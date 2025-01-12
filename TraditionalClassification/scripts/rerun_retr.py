import os
import json
import time
import re
from retrieval import Retriever, AudioDataset, DataLoader
from utils.gpu import find_free_gpu


def get_model_name(name):
    from model import SimpleCNN, SimpleResnet, SimpleLSTM, CRNN, AttentionCNN

    name = name.lower()
    if "resnet" in name:
        return SimpleResnet
    if "lstm" in name:
        return SimpleLSTM
    if "crnn" in name:
        return CRNN
    if "attention" in name:
        return AttentionCNN
    return SimpleCNN


def extract_log_directory(log_dir, clean_func=None):
    if not os.path.isdir(log_dir):
        print(f"目录 {log_dir} 不存在！")
        return
    if clean_func:
        clean_func(log_dir)

    results = []
    for sub_dir in os.listdir(log_dir):
        sub_dir_path = os.path.join(log_dir, sub_dir)

        if os.path.isdir(sub_dir_path):
            if not enough_rec(sub_dir_path):
                results.append(sub_dir_path)
                continue
    return results


def extract_summary(log_dir):
    # 确保 log_dir 是存在的
    if not os.path.isdir(log_dir):
        print(f"目录 {log_dir} 不存在！")
        return
    retrieval_res = None
    acc = 0
    config = {}
    method = ""
    for file in os.listdir(log_dir):
        if file == "evaluation_results.json":
            with open(os.path.join(log_dir, file), "r") as f:
                results = json.load(f)
            retrieval_res = results["result"]
            config.update(results["config"])
        if file == "config.json":
            with open(os.path.join(log_dir, file), "r") as f:
                cfg = json.load(f)
            config.update(cfg)

    # for file in os.listdir(log_dir):
    # method = config["method"]
    # pattern = f"{method}" + r"_\d+\.\d?"
    # match = re.search(pattern, file)
    # if match:
    #     acc = float(file.split("_")[-1])
    #     break

    return {
        "config": config,
        "retrieval_res": retrieval_res,
        # "acc": acc,
        "path": log_dir,
    }


def enough_rec(log_dir):
    # 确保 log_dir 是存在的
    if not os.path.isdir(log_dir):
        print(f"目录 {log_dir} 不存在！")
        return

    exist_model = False
    enough = False
    for file in os.listdir(log_dir):
        if file.endswith(".pth"):
            exist_model = True
            break
        if file == "retrieval_results.jsonl":
            with open(os.path.join(log_dir, file), "r") as f:
                for line in f:
                    qa = json.loads(line)
                    if len(list(qa.values())[0]) >= 20:
                        enough = True

    if exist_model and not enough:
        return False
    return True


def main():
    log_dir = "checkpoints"
    rerun_dirs = extract_log_directory(log_dir)
    cache_dataloaders = {}
    gpu_id = 0
    for dir in rerun_dirs:
        dir_path = dir
        print(f"Rerun {dir_path}")
        config = extract_summary(dir_path)["config"]
        model_name = get_model_name(config["model"])
        if config["model"] == "lstm":
            input_dim = 0
            if config["method"] == "stft":
                input_dim = config["window_length"] // 2 + 1
            if "deravative" in config["method"]:
                input_dim = 39
            if "mfcc" in config["method"]:
                input_dim = 13
            model = model_name(input_dim=input_dim)
        else:
            model = model_name()

        # while True:
        # gpu_ids = find_free_gpu()
        # if gpu_id not in gpu_ids:
        #     time.sleep(10)
        #     continue
        # if gpu_ids:
        #     if gpu_id is None or gpu_id not in gpu_ids:
        #         gpu_id = gpu_ids[-1]
        #     break
        device = f"cuda:{gpu_id}"
        print("device", device)
        retriever = Retriever(model, device)

        cache_config = (config["window_length"], config["method"], config["transform"])
        if cache_config in cache_dataloaders:
            train_dataloader, test_dataloader = cache_dataloaders[cache_config]
        else:
            train_dataset = AudioDataset(
                feature_type=config["method"],
                config=config,
                fold="train",
            )
            test_dataset = AudioDataset(
                feature_type=config["method"],
                config=config,
                fold="test",
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"] // 2,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=config["batch_size"] // 2,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            cache_dataloaders[cache_config] = train_dataloader, test_dataloader

        res = retriever.generate_retrieval_results(
            test_dataloader, train_dataloader, device
        )
        retriever.save_results(res, os.path.join(dir_path, "retrieval_results.jsonl"))
        ans = retriever.calculate_metric(
            os.path.join(dir_path, "retrieval_results.jsonl")
        )

        cur_ans = {}

        with open(os.path.join(dir_path, "evaluation_results.json"), "r") as f:
            cur_ans = json.load(f)
        with open(os.path.join(dir_path, "evaluation_results.json"), "w") as f:
            cur_ans["result"] = ans
            json.dump(cur_ans, f)


if __name__ == "__main__":
    main()
