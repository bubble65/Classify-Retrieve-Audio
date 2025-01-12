import os
import json
import logging
import torch
import random
import argparse
import torch.nn as nn
import torchaudio
import numpy as np

from time import sleep
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional, List
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics.pairwise import cosine_similarity

from utils.gpu import find_free_gpu
from numpy.fft import fft
from method.mfcc import (
    compute_mfcc,
    compute_mel_spectrogram,
    compute_stft_DCT,
    compute_deltas,
    compute_mfcc_with_derivatives,
)
from method.stft import (
    compute_stft,
    # stand_stft as compute_stft,
)

from model import SimpleResnet, SimpleLSTM, SimpleCNN, AttentionCNN, CRNN
from evaluate_results import evaluate_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--method", type=str, default="mfcc")
parser.add_argument("--model", type=str, default="resnet")
parser.add_argument("--window_length", type=int, default=1024)
parser.add_argument("--hop_length", type=int, default=None)
parser.add_argument("--transform", type=str, default="dct")
args = parser.parse_args()


cur_time = datetime.strftime(datetime.now(), "%m-%d_%H:%M:%S")
checkpoint_dir = (
    f"checkpoints/{cur_time}_{args.model}_{args.method}_{args.window_length}"
)
os.mkdir(checkpoint_dir)
checkpoint_dir = Path(checkpoint_dir, exist_ok=True)

# 设置日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 创建控制台处理器并设置日志级别为DEBUG
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# 创建文件处理器并设置日志级别为DEBUG
fh = logging.FileHandler(checkpoint_dir / "training.log")
fh.setLevel(logging.DEBUG)

# 创建格式化器并将其添加到处理器
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# 将处理器添加到日志记录器
logger.addHandler(ch)
logger.addHandler(fh)


class AudioDataset(Dataset):
    def __init__(
        self,
        root_dir="data/",
        fold=None,
        feature_type="mfcc",
        config=None,
        cache_dir="cache/features_cache",
        num_workers=4,
        cache: bool = True,
        derivative: bool = False,
    ):
        self.root_dir = root_dir
        self.fold = fold
        self.feature_type = feature_type
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # 设置默认配置
        default_config = {
            "window_length": 2048,
            "hop_length": 512,
            "n_mfcc": 13,
            "n_mels": 40,
            "window_type": "hann",
        }
        self.config = default_config if config is None else {**default_config, **config}

        self.file_list, self.labels = self._load_file_list()
        config_str = "_".join([f"{k}_{v}" for k, v in self.config.items()])

        self.features_cache_file = (
            self.cache_dir / f"{feature_type}_{config_str}_features_{fold}.npy"
        )
        self.labels_cache_file = (
            self.cache_dir / f"{feature_type}_{config_str}_labels_{fold}.npy"
        )

        if "stft" in self.feature_type:
            # self.feature_type = self.feature_type.split("_")[0]
            cache = False

        if cache:
            if self.features_cache_file.exists() and self.labels_cache_file.exists():
                print(f"Loading cached features from {self.features_cache_file}")
                self.features = self._load_cached_features(self.features_cache_file)
                self.labels = self._load_cached_features(self.labels_cache_file)
                print(f"Loaded {len(self)} samples")
            else:
                if not self.features_cache_file.exists():
                    self.features, self.labels = self._cal_features()
                    np.save(self.features_cache_file, self.features)
                if not self.labels_cache_file.exists():
                    self.labels = self._cal_labels()
                    np.save(self.labels_cache_file, self.labels)
        else:
            self.features, self.labels = self._cal_features()

        config_repr = "\n".join([f"{k}: {v}" for k, v in self.config.items()])
        logger.info(f"Current config:\n{config_repr}")
        logger.info(f"Feature_shape: {self.features.shape}")

    def _load_file_list(self):
        """加载文件列表并解析标签"""
        file_list = []
        labels = []
        for file in os.listdir(self.root_dir):
            if file.endswith(".wav"):
                fold_num = int(file.split("-")[0])
                if self.fold == "train" and fold_num < 5:
                    file_list.append(os.path.join(self.root_dir, file))
                    labels.append(int(file.split("-")[-1].split(".")[0]))
                elif self.fold == "test" and fold_num == 5:
                    file_list.append(os.path.join(self.root_dir, file))
                    labels.append(int(file.split("-")[-1].split(".")[0]))
        return file_list, labels

    def _load_cached_features(self, file_path):
        return np.load(file_path)

    def _cal_labels(self):
        """计算所有标签"""
        labels = []
        for file in self.file_list:
            label = int(file.split("-")[-1].split(".")[0])
            labels.append(label)
        return labels

    def _cal_features(self):
        waveform, sr = torchaudio.load(self.file_list[0])
        first_feature = self.extract_features(waveform, sr)
        features_shape = (len(self.file_list),) + tuple(first_feature.shape)

        all_features = np.zeros(features_shape, dtype=np.float32)
        all_labels = np.array(self.labels, dtype=np.int64)

        for idx in tqdm(
            range(len(self.file_list)), desc=f"Extracting features {self.feature_type}"
        ):
            try:
                waveform, sr = torchaudio.load(self.file_list[idx])
                features = self.extract_features(waveform, sr)
                all_features[idx] = features.numpy()
            except Exception as e:
                print(f"Error processing file {self.file_list[idx]}: {str(e)}")
                raise
        if "stft" in self.feature_type:
            print(f"【{self.feature_type}】Features shape: {all_features.shape}")
        return all_features, all_labels

    def extract_fft_features(self, waveform: np.ndarray) -> np.ndarray:
        """提取 FFT 特征"""
        # 对每个窗口进行 FFT
        n_frames = (
            1
            + (len(waveform) - self.config["window_length"])
            // self.config["hop_length"]
        )
        frames = np.zeros((n_frames, self.config["window_length"]))

        for i in range(n_frames):
            start = i * self.config["hop_length"]
            end = start + self.config["window_length"]
            frame = waveform[start:end]
            frames[i] = np.abs(fft(frame))  # 取幅度谱

        return 20 * np.log10(frames + 1e-10)  # 转换为分贝刻度

    def extract_features(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        """
        从波形数据中提取特征

        Args:
            waveform: 输入波形，形状为 [channels, samples]
            sample_rate: 采样率

        Returns:
            特征张量，形状为 [1, time_steps, features]
        """
        # 确保输入是单声道
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        if waveform.ndim == 2:
            waveform = waveform[0]  # 取第一个通道

        # 根据特征类型选择相应的提取方法
        if self.feature_type == "raw":
            features = waveform[np.newaxis, :].reshape(-1, sample_rate)

        elif self.feature_type == "fft":
            features = np.fft.rfft(waveform)  # shape = (samples,)
            features = np.abs(features)
            features = features[np.newaxis, :].reshape(
                1, -1
            )  # shape = (-1, samples_rate)
            # print(features)
            features = 20 * np.log10(features + 1e-10)

        elif self.feature_type == "stft":
            features, _, _ = compute_stft(
                waveform,
                window_length=self.config["window_length"],
                hop_length=self.config["hop_length"],
                window_type=self.config["window_type"],
            )
            features = 20 * np.log10(np.abs(features) + 1e-10)

        elif self.feature_type == "stft_derivatives":
            features, _, _ = compute_stft(
                waveform,
                window_length=self.config["window_length"],
                hop_length=self.config["hop_length"],
                window_type=self.config["window_type"],
            )

            delta = compute_deltas(features)
            delta2 = compute_deltas(delta)
            features = np.hstack([features, delta, delta2])
            features = 20 * np.log10(np.abs(features) + 1e-10)

        elif self.feature_type == "stft_DCT":
            features, _, _ = compute_stft_DCT(
                waveform,
                sample_rate=sample_rate,
                n_mfcc=self.config["n_mfcc"],
                n_mels=self.config["n_mels"],
                window_length=self.config["window_length"],
                hop_length=self.config["hop_length"],
                window_type=self.config["window_type"],
            )
            features = features.T

        elif self.feature_type == "mfcc":
            features, _, _ = compute_mfcc(
                waveform,
                sample_rate=sample_rate,
                n_mfcc=self.config["n_mfcc"],
                n_mels=self.config["n_mels"],
                window_length=self.config["window_length"],
                hop_length=self.config["hop_length"],
                transform=self.config["transform"],
            )
            features = features.T

        elif self.feature_type == "mfcc_derivatives":
            features = compute_mfcc_with_derivatives(
                waveform,
                sample_rate=sample_rate,
                n_mfcc=self.config["n_mfcc"],
                n_mels=self.config["n_mels"],
                window_length=self.config["window_length"],
                hop_length=self.config["hop_length"],
                transform=self.config["transform"],
                gradient=2,
            )
            features = features.T

        elif self.feature_type == "mel":
            features, _, _ = compute_mel_spectrogram(
                waveform,
                sample_rate,
                n_mels=self.config["n_mels"],
                window_length=self.config["window_length"],
                hop_length=self.config["hop_length"],
            )

        else:
            raise ValueError(f"不支持的特征类型: {self.feature_type}")

        # 标准化特征
        features = (features - features.mean()) / (features.std() + 1e-10)

        # 转换为张量并添加通道维度
        features = torch.from_numpy(features).float()
        features = features.unsqueeze(0)  # [1, time_steps, features]

        return features

    def _get_feature_shape(self):
        """获取特征形状以创建缓存数据集"""
        waveform, sr = torchaudio.load(self.file_list[0])
        features = self.extract_features(waveform, sr)
        return features.shape

    def __getitem__(self, idx):
        """获取预计算的特征和标签"""
        features = torch.from_numpy(self.features[idx]).float()
        label = torch.tensor(self.labels[idx]).long()
        return {
            "features": features,
            "label": label,
            "path": self.file_list[idx],
        }

    def __len__(self):
        return len(self.file_list)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=10,
    device="cuda",
    early_stopping=5,
    method=None,
):
    model = model.to(device)
    model_name = model.__class__.__name__
    best_acc = 0.0
    cur_count = 0

    for epoch in tqdm(range(num_epochs), desc=f"{model_name} on {method}"):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            inputs, labels = batch["features"].to(device), batch["label"].to(device)
            # logger.debug(f"[Training]: inputs: {inputs.shape}, labels: {labels.shape}")

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if scheduler:
                scheduler.step()

        train_acc = 100.0 * correct / total

        # Validation phase
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch["features"].to(device), batch["label"].to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * correct / total
        logger.debug(
            f"Epoch [{epoch+1}/{num_epochs}] of [{model_name}] | [{method}] Loss: {running_loss} Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            cur_count = 0
            logger.info(
                f"Saving best model {model_name}_{method} with val acc {val_acc:.2f}%"
            )
            torch.save(
                model.state_dict(),
                checkpoint_dir / f"best_model_{model_name}_{method}.pth",
            )
        else:
            cur_count += 1
        if cur_count >= early_stopping:
            logger.info(
                f"Early stopping at epoch {epoch+1} for model {model_name} at val acc {best_acc:.2f}%"
            )
            break


def create_optimized_dataloaders(
    feature_type="mfcc", batch_size=32, num_workers=4, pin_memory=True, config={}
):
    train_dataset = AudioDataset(
        feature_type=feature_type,
        fold="train",
        config=config,
        cache_dir=f"cache/features_cache_{feature_type}",
        cache=False,
    )

    test_dataset = AudioDataset(
        feature_type=feature_type,
        fold="test",
        config=config,
        cache_dir=f"cache/features_cache_{feature_type}",
        cache=False,
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def create_optimized_datasets(feature_type="mfcc", config={}):
    train_dataset = AudioDataset(
        feature_type=feature_type,
        fold="train",
        config=config,
        cache_dir=f"cache/features_cache_{feature_type}",
        # cache=False,
    )

    test_dataset = AudioDataset(
        feature_type=feature_type,
        fold="test",
        config=config,
        cache_dir=f"cache/features_cache_{feature_type}",
        # cache=False,
    )

    return train_dataset, test_dataset


def evaluate_model(
    model, test_loader, device="cuda", checkpoint_d: Path = None, method=None
):
    model_name = model.__class__.__name__
    if checkpoint_d is None:
        raise ValueError("checkpoint_d must be provided")

    model.load_state_dict(
        torch.load(
            checkpoint_d / f"best_model_{model_name}_{method}.pth", map_location=device
        )
    )
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch["features"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    logger.info(f"Evaluating: {model_name}_{method}: Test accuracy: {acc:.2f}%")
    return acc


def train_eval(args, config) -> Dict:
    """
    return: model, train_loader, test_loader
    """
    res_logger = logging.getLogger("res")
    res_logger.setLevel(logging.DEBUG)
    res_fh = logging.FileHandler(f"results/res_{args.method}.log")
    res_fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    res_fh.setFormatter(formatter)
    res_logger.addHandler(res_fh)

    train_dataset, test_dateset = create_optimized_datasets(
        feature_type=args.method, config=config
    )

    sleep(random.uniform(0, 20))
    gpu_ids = find_free_gpu()
    while not gpu_ids:
        sleep(30)
        gpu_ids = find_free_gpu()
    print("gpu_ids", gpu_ids)
    gpu_id = gpu_ids[-1]
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    if gpu_id in ["1", "3"]:
        batch_size = args.batch_size // 4
    else:
        batch_size = args.batch_size

    learning_rate = 0.001
    num_epochs = 1000
    method = args.method

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dateset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if args.model == "lstm":
        input_dim = train_dataset[0]["features"].shape[-1]
        model = SimpleLSTM(
            input_dim=input_dim, hidden_dim=128, num_layers=2, num_classes=10
        )
    else:
        model = {
            "resnet": SimpleResnet,
            "lstm": SimpleLSTM,
            "cnn": SimpleCNN,
            "crnn": CRNN,
            "attention_cnn": AttentionCNN,
        }[args.model](num_classes=50)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    logger.info(
        "Training args:\n"
        f"model: {model.__class__.__name__}\n"
        f"batch_size: {batch_size}\n"
        f"learning_rate: {learning_rate}\n"
        f"num_epochs: {num_epochs}\n"
        f"device: {device}\n"
        f"method: {method}\n"
    )

    train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=num_epochs,
        device=device,
        early_stopping=30,
        method=method,
    )

    acc = evaluate_model(
        model, test_loader, device=device, checkpoint_d=checkpoint_dir, method=method
    )
    with open(checkpoint_dir / f"{model.__class__.__name__}_{method}_{acc}", "w") as f:
        json.dump(config, f, indent=4)

    # content = ""
    # cur_res = {}
    # with open(f"results/res_{args.method}.json", "w+") as f:
    #     content = f.read()
    #     if content:
    #         cur_res = json.loads(content)

    # with open(f"results/res_{args.method}.json", "w") as f:
    #     info = {
    #         "config": config,
    #         "acc": acc,
    #     }
    #     cur_res.update({checkpoint_dir.name: info})
    #     json.dump(cur_res, f, indent=4)

    config_str = "\n".join([f"{k}: {v}" for k, v in config.items()])
    res_logger.info(
        f"\n{config_str}" + f"\n{cur_time}" + f"\nTest accuracy: {acc:.2f}%"
    )
    logger.info(f"Test accuracy: {acc:.2f}%")
    return {
        "model": model,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "device": device,
    }


def generate_retrieval_results(
    model: nn.Module,
    query_dataloader: DataLoader,
    database_dataloader: DataLoader,
    device: str,
    top_k: int = 20,
) -> Dict[str, List[str]]:
    """Generate retrieval results using the datasets.

    Args:
        model: Neural network model for feature extraction
        query_dataset: Dataset containing query audio files
        database_dataset: Dataset containing database audio files
        top_k: Number of top results to return per query

    Returns:
        Dictionary mapping query filenames to lists of top-k matching database filenames
    """
    results = {}
    model = model.to(device=device)
    model.eval()

    # Extract features for database files
    print("Extracting features for database files...")
    database_features = {}
    with torch.no_grad():
        for batch in tqdm(database_dataloader):
            file_list = batch["path"]
            hidden_features = model.extract_features(batch["features"].to(device))
            hidden_features = hidden_features.cpu()
            # logger.debug(f"hidden_features: {hidden_features.shape}")
            database_features.update(
                {file: feat for file, feat in zip(file_list, hidden_features)}
            )

    # Process each query
    print("Processing queries...")
    with torch.no_grad():
        for batch in tqdm(query_dataloader):
            file_list = batch["path"]
            query_features = model.extract_features(batch["features"].to(device))
            query_features = query_features.cpu()

            for i, query_file in enumerate(file_list):
                # Get current query features
                curr_query_features = query_features[i]

                # Compute similarities with all database items
                similarities = []
                for db_file, db_features in database_features.items():
                    # logger.debug(f"curr_query_features: {curr_query_features.shape}")
                    # logger.debug(f"db_features: {db_features.shape}")
                    sim = torch.nn.functional.cosine_similarity(
                        curr_query_features.unsqueeze(0), db_features.unsqueeze(0)
                    ).item()
                    similarities.append((db_file, sim))

                # Sort by similarity and get top K results
                similarities.sort(key=lambda x: x[1], reverse=True)
                # Convert paths to relative format for the results
                results[Path(query_file).name] = [
                    Path(file).name for file, _ in similarities[:top_k]
                ]

    return results


def write_jsonl(ls, path):
    with open(path, "w") as f:
        for item in ls:
            f.write(json.dumps(item) + "\n")


def main():
    global args
    if not args.hop_length:
        args.hop_length = args.window_length // 4

    default_config = {
        "window_length": 2048,
        "hop_length": 512,
        "n_mfcc": 13,
        "n_mels": 40,
        "window_type": "hann",
        "model": "resnet",
    }
    new_config = {
        "window_length": args.window_length,
        "hop_length": args.hop_length,
        "transform": args.transform,
        "model": args.model,
    }

    config = {**default_config, **new_config}
    log_config = {**config, **args.__dict__}

    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(log_config, f, indent=4)
    try:
        model, train_loader, test_loader, device = train_eval(args, config).values()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.error("CUDA out of memory error. Retrying...")
            torch.cuda.empty_cache()
            sleep(10)
            model, train_loader, test_loader, device = train_eval(args, config).values()
        else:
            raise e
    results = generate_retrieval_results(model, test_loader, train_loader, device)
    results = [{k: v} for k, v in results.items()]
    write_jsonl(results, checkpoint_dir / "retrieval_results.jsonl")

    # Evaluate results
    result_file = checkpoint_dir / "retrieval_results.jsonl"
    res = evaluate_metrics(result_file)
    with open(checkpoint_dir / "evaluation_results.json", "w") as f:
        d = {"result": res, "config": log_config, "dir": str(checkpoint_dir)}
        json.dump(d, f, indent=4)

    retr_res_dir = Path("retrieval_res")

    with open(retr_res_dir / f"{args.model}_{args.method}_res.json", "w") as f:
        d = {"result": res, "config": log_config, "dir": str(checkpoint_dir)}
        json.dump(d, f, indent=4)

    print(res)


if __name__ == "__main__":
    # models = ['cnn', 'crnn', 'attention_cnn', 'lstm', 'resnet']
    # window_lengths = ['1024', '2048', '4096']

    # # 定义不同方法的配置
    # configurations = [
    #     {'method': 'raw', 'batch_size': '128'},
    #     {'method': 'fft', 'batch_size': '200'},
    #     {'method': 'mel', 'batch_size': '1500'},
    #     {'method': 'mfcc', 'batch_size': '100', 'transform': 'dst'},
    #     {'method': 'mfcc', 'batch_size': '100', 'transform': 'dct'},
    #     {'method': 'mfcc_derivatives', 'batch_size': '100', 'transform': 'dst'},
    #     {'method': 'mfcc_derivatives', 'batch_size': '100', 'transform': 'dct'},
    #     {'method': 'stft', 'batch_size': '100'},
    #     {'method': 'stft_DCT', 'batch_size': '100'},
    #     {'method': 'stft_derivatives', 'batch_size': '100'}
    # ]
    # for cfg in configurations:
    #     for model in models:
    #         for window_length in window_lengths:
    #             args = argparse.Namespace(
    #                 batch_size=int(cfg['batch_size']),
    #                 method=cfg['method'],
    #                 model=model,
    #                 window_length=int(window_length),
    #                 hop_length=None,
    #                 transform=cfg.get('transform', 'dst')
    #             )
    # train_dataset, test_dateset = create_optimized_datasets(
    #     feature_type=args.method
    # )
    main()
