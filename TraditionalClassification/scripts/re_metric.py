import os
import json
from tqdm import tqdm
import time
import re
from retrieval import Retriever, AudioDataset, DataLoader
from utils.gpu import find_free_gpu


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
            if enough_rec(sub_dir_path):
                results.append(sub_dir_path)
                continue
    return results


def extract_summary(log_dir):
    # 确保 log_dir 是存在的
    if not os.path.isdir(log_dir):
        print(f"目录 {log_dir} 不存在！")
        return
    retrieval_res = None
    config = {}
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
    if "lstm" in log_dir:
        return False
    for file in os.listdir(log_dir):
        if file == "retrieval_results.jsonl":
            with open(os.path.join(log_dir, file), "r") as f:
                for line in f:
                    qa = json.loads(line)
                    if len(list(qa.values())[0]) >= 20:
                        return True

    return False


def main():
    log_dir = "checkpoints"
    rerun_dirs = extract_log_directory(log_dir)
    for dir in tqdm(rerun_dirs):
        dir_path = dir
        print(f"Rerun {dir_path}")
        config = extract_summary(dir_path)["config"]

        ans = Retriever.calculate_metric(
            os.path.join(dir_path, "retrieval_results.jsonl")
        )
        print(ans)

        with open(os.path.join(dir_path, "evaluation_results.json"), "w") as f:
            json.dump(
                {
                    "config": config,
                    "result": ans,
                    "path": dir_path,
                },
                f,
                indent=4,
            )


if __name__ == "__main__":
    main()
