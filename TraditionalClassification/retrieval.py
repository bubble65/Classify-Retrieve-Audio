import os
import torch
import torch.nn as nn
import torchaudio
import random
import numpy as np
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from utils.gpu import find_free_gpu
from numpy.fft import fft
from method.mfcc import (
    compute_mfcc,
    # stand_mfcc as compute_mfcc,
    compute_mel_spectrogram,
    # stand_mel as compute_mel_spectrogram,
    compute_stft_DCT,
    compute_deltas,
    compute_mfcc_with_derivatives,
)
from method.stft import (
    compute_stft,
    # stand_stft as compute_stft,
)

# from main import AudioDataset

from evaluate_results import evaluate_metrics

time.sleep(random.uniform(0, 2))


# 设置日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 创建控制台处理器并设置日志级别为DEBUG
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# 创建文件处理器并设置日志级别为DEBUG
fh = logging.FileHandler(f"retrieval_{random.uniform(0, 10)}.log")
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
                logger.info(f"Loading cached features from {self.features_cache_file}")
                self.features = self._load_cached_features(self.features_cache_file)
                self.labels = self._load_cached_features(self.labels_cache_file)
                logger.info(f"Loaded {len(self)} samples")
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
            ).T

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


def extract_features(model: nn.Module, dataset_item: Dict) -> np.ndarray:
    """Extract features from audio using ResNet model."""
    features = dataset_item["features"].unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        # Get features from the layer before classification
        features = model.extract_features(features)
        # print(f"features shape: {features.shape}")

    return features.squeeze().cpu().numpy()


def compute_similarity(
    query_features: np.ndarray, candidate_features: np.ndarray
) -> float:
    """Compute cosine similarity between feature vectors."""
    return np.dot(query_features, candidate_features) / (
        np.linalg.norm(query_features) * np.linalg.norm(candidate_features)
    )


class Retriever:
    def __init__(self, model: nn.Module = None, device="cuda"):
        if model:
            self.model = model.to(device)
        else:
            self.model = model
        self.device = device

    def generate_retrieval_results(
        self,
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
        device = self.device
        self.model.eval()

        print("Extracting features for database files...")
        database_features = {}
        with torch.no_grad():
            for batch in tqdm(database_dataloader):
                file_list = batch["path"]
                features = batch["features"].to(device)

                hidden_features = self.model.extract_features(features)

                hidden_features = hidden_features.cpu()
                database_features.update(
                    {file: feat for file, feat in zip(file_list, hidden_features)}
                )

        db_files = list(database_features.keys())
        db_tensor = torch.stack([database_features[file] for file in db_files])
        # Normalize database features once
        db_tensor = torch.nn.functional.normalize(db_tensor, p=2, dim=1)
        db_tensor = db_tensor.to(device)

        print("Processing queries...")
        with torch.no_grad():
            for batch in tqdm(query_dataloader):
                file_list = batch["path"]
                features = batch["features"].to(device)
                query_features = self.model.extract_features(features)
                query_features = query_features.cpu()

                # Extract and normalize query features
                query_features = self.model.extract_features(features)

                query_features = torch.nn.functional.normalize(
                    query_features, p=2, dim=1
                )

                # Compute similarities for entire batch at once
                # Shape: (batch_size, num_database_items)
                similarities = torch.mm(query_features, db_tensor.t())

                # Get top-k indices for each query in batch
                _, top_indices = torch.topk(similarities, k=top_k, dim=1)

                # Process results for each query in batch
                for i, query_file in enumerate(file_list):
                    top_files = [db_files[idx] for idx in top_indices[i].cpu()]
                    results[Path(query_file).name] = [
                        Path(file).name for file in top_files
                    ]

                # Clear GPU memory
                del query_features
                torch.cuda.empty_cache()
        return results

    def save_results(self, results: Dict[str, List[str]], output_path: str):
        """Save results in JSONL format."""
        with open(output_path, "w") as f:
            for q, a in results.items():
                f.write(json.dumps({q: a}) + "\n")

    @staticmethod
    def calculate_metric(result_file: str, top_k_list: list[int] = [10, 20]):
        """Evaluate retrieval results."""
        return evaluate_metrics(result_file, top_k_list)


def main(model: nn.Module, cfg: dict, device):
    # Set model to evaluation mode
    model = model.to(device)
    print("device: ", device)
    print("model : ", next(model.parameters()).device)
    model.eval()
    feature_type = cfg["method"]

    # Create datasets
    query_dataset = AudioDataset(
        root_dir="data/",
        fold="test",  # Use fold 5 for queries
        feature_type=feature_type,
        config=cfg,
        cache_dir=f"cache/features_cache_{feature_type}",
        cache=False,
    )

    database_dataset = AudioDataset(
        root_dir="data/",
        fold="train",  # Use folds 1-4 for database
        feature_type=feature_type,
        config=cfg,
        cache_dir=f"cache/features_cache_{feature_type}",
        cache=False,
    )
    batch_size = cfg.get("batch_size", 100)
    query_dataloader = DataLoader(
        query_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    database_dataloader = DataLoader(
        database_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Generate retrieval results
    results = generate_retrieval_results(
        model, query_dataloader, database_dataloader, device=device, top_k=20
    )

    # Save results
    ckpt_dir = cfg["dir"]
    cfg.pop("dir")
    save_results(results, ckpt_dir / "retrieval_results.jsonl")

    # Print example result
    example_query = next(iter(results.keys()))
    print("\nExample retrieval result:")
    print(f"Query: {example_query}")
    print(f"Top 20 matches: {results[example_query]}")

    # Evaluate results
    result_file = ckpt_dir / "retrieval_results.jsonl"
    res = evaluate_metrics(result_file, top_k=20)
    print("#" * 40, "res is ok!")
    print("#" * 40, str(ckpt_dir))
    with open(ckpt_dir / "evaluation_results.json", "w") as f:
        d = {"result": res, "config": cfg, "dir": str(ckpt_dir)}
        json.dump(d, f, indent=4)
    print("#" * 40, "dump is ok!")

    retr_res_dir = Path("retrieval_res")

    with open(retr_res_dir / f"{ckpt_dir.name}_res.json", "w") as f:
        d = {"result": res, "config": cfg, "dir": str(ckpt_dir)}
        json.dump(d, f, indent=4)

    print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Retrieval System")
    # parser.add_argument("--feature_type", type=str, default="stft", help="Type of audio features to use")
    parser.add_argument(
        "--checkpoint_time",
        type=str,
        required=True,
        help="Checkpoint time to load the model",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name to load"
    )
    parser.add_argument("--input_dim", type=int, help="Model name to load", default=13)
    args = parser.parse_args()
    from model import SimpleResnet, AttentionCNN, CRNN, SimpleCNN, SimpleLSTM
    import re

    model_name = args.model_name
    ck_time = args.checkpoint_time
    tasks = []

    print(ck_time)
    print(f"model: {model_name}")
    ckpt_dir = Path(f"{ck_time}")
    ckpt_files = list(ckpt_dir.glob("*.pth"))
    with open(ckpt_dir / "config.json", "r") as f:
        cfg = json.load(f)
    cfg["dir"] = ckpt_dir
    model = {
        "SimpleResnet": SimpleResnet,
        "AttentionCNN": AttentionCNN,
        "CRNN": CRNN,
        "SimpleCNN": SimpleCNN,
        "SimpleLSTM": SimpleLSTM,
    }[model_name](num_classes=50)
    # if model_name == "SimpleLSTM":
    # {
    #     'mfcc': 13,
    #     'mel': 40,
    #     'stft': 513,
    # }
    # if cfg['method'] == 'mfcc':
    #     input_dim = 13
    # elif cfg['method'] == 'mel':
    #     input_dim = 40
    # elif cfg
    # model = model(input_dim=args.input_dim, hidden_dim=128, num_layers=2, num_classes=50)
    # else:
    #     model = model(num_classes=50)

    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
    gpu_ids = find_free_gpu()
    while not gpu_ids:
        time.sleep(20)
        gpu_ids = find_free_gpu()
    gpu_id = gpu_ids[-1]
    device = torch.device(f"cuda:{gpu_id}")
    print("Current device:", device)
    model.load_state_dict(torch.load(ckpt_files[0], map_location=device))
    main(model, cfg, device=device)
