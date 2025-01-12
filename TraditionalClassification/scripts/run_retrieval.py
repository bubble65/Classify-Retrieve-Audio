# traversal.py
import os
import re
import gc
import time
import json
from pathlib import Path
import torch
import subprocess
from typing import List, Tuple
import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
from utils.gpu import find_free_gpu


def get_tasks() -> List[Tuple[str, str]]:
    """
    扫描checkpoints目录，返回待处理的任务列表
    返回: [(checkpoint_path, model_name), ...]
    """
    log_dir = "checkpoints"
    tasks = []

    for sub_dir in sorted(os.listdir(log_dir)):
        sub_dir_path = os.path.join(log_dir, sub_dir)
        has_model = False
        finished = False
        model_name = None

        for fl in os.listdir(sub_dir_path):
            if fl == "evaluation_results.json":
                finished = True
            if fl.endswith(".pth"):
                match = re.match(r"best_model_([^\W_]+)_(\w+).pth$", fl)
                if match:
                    model_name = match.group(1)
                has_model = True

        if has_model and not finished:
            tasks.append((sub_dir_path, model_name))

    return tasks


def run_evaluation(task: Tuple[str, str]) -> None:
    """
    运行单个评估任务的函数
    Args:
        task: (checkpoint_path, model_name)
    """
    ckpt_path, model_name = task

    try:
        cmd = [
            "python",
            "retrieval.py",
            "--checkpoint_time",
            ckpt_path,
            "--model_name",
            model_name,
        ]

        logging.info(f"Starting evaluation for {ckpt_path}")
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )

        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logging.error(f"Error in evaluation for {ckpt_path}")
            logging.error(f"stdout: {stdout}")
            logging.error(f"stderr: {stderr}")
        else:
            logging.info(f"Successfully completed evaluation for {ckpt_path}")

    except Exception as e:
        logging.error(f"Exception occurred while processing {ckpt_path}: {str(e)}")

    finally:
        time.sleep(5)


def main():
    tasks = get_tasks()
    logging.info(f"Found {len(tasks)} tasks to process")

    # 确定要使用的最大进程数
    max_workers = min(len(find_free_gpu()), 2)
    if max_workers == 0:
        max_workers = 1

    logging.info(f"Starting evaluation with {max_workers} workers")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(run_evaluation, tasks)


if __name__ == "__main__":
    main()
