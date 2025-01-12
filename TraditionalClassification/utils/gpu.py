import os
import subprocess
from typing import Union, List, Dict


def get_gpu_info() -> List[Dict[str, Union[int, float, str]]]:
    """
    获取所有GPU的信息，包括显存使用率和计算利用率

    Returns:
        List[Dict]: 包含每个GPU信息的字典列表
    """
    try:
        # 使用nvidia-smi命令获取GPU信息
        nvidia_smi = "nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits"
        output = subprocess.check_output(nvidia_smi.split(), universal_newlines=True)

        gpu_info = []
        for line in output.strip().split("\n"):
            index, total_memory, used_memory, free_memory, gpu_util, temp = map(
                float, line.split(",")
            )

            gpu_info.append(
                {
                    "index": int(index),
                    "total_memory": int(total_memory),
                    "used_memory": int(used_memory),
                    "free_memory": int(free_memory),
                    "gpu_util": int(gpu_util),
                    "temperature": int(temp),
                }
            )

        return gpu_info

    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("无法获取GPU信息。请确保已安装NVIDIA驱动且nvidia-smi可用。")


def find_free_gpu_by_resource():
    gpu_info = get_gpu_info()

    best_gpu = []
    for gpu in gpu_info:
        if gpu["used_memory"] == 0 and gpu["gpu_util"] == 0:
            best_gpu.append((gpu['index'], gpu["used_memory"], gpu["gpu_util"]))

    best_gpu.sort(key=lambda x: (x[1], x[2]))
    return [idx for idx, _, _ in best_gpu]


def get_gpu_processes() -> List[Dict[str, str]]:
    """
    获取当前所有 GPU 上运行的进程信息。

    Returns:
        List[Dict]: 每个字典包含 GPU ID、用户、进程 ID、显存占用等信息。
    """
    try:
        # 使用 nvidia-smi 获取进程信息
        command = "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits"
        output = subprocess.check_output(command.split(), universal_newlines=True)

        processes = []
        for line in output.strip().split("\n"):
            gpu_uuid, pid, process_name, used_memory = line.split(", ")
            processes.append({
                "gpu_uuid": gpu_uuid,
                "pid": pid,
                "process_name": process_name,
                "used_memory": used_memory
            })

        return processes

    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("无法获取 GPU 进程信息。请确保已安装 NVIDIA 驱动且 nvidia-smi 可用。")


def find_free_gpu():
    processes = get_gpu_processes()
    # 找到所有被占用的 GPU UUID
    busy_gpus = {proc["gpu_uuid"] for proc in processes}
    # 使用 nvidia-smi 获取所有 GPU 信息
    command = "nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader"
    all_gpus = subprocess.check_output(command.split(), universal_newlines=True).strip().split("\n")
    uuid2id = {uuid: i for i, uuid in enumerate(all_gpus)}
    # 筛选未被使用的 GPU
    free_gpus = [gpu for gpu in all_gpus if gpu not in busy_gpus]
    free_gpus = [uuid2id[uuid] for uuid in free_gpus]
    A800 = [gpu for gpu in free_gpus if gpu not in [1, 3]]
    Rtx3090 = [gpu for gpu in free_gpus if gpu not in A800]
    # 返回空闲 GPU 的 ID
    return Rtx3090 + A800


def set_gpu_environment(gpu_idx: List[int]) -> None:
    """
    设置CUDA环境变量，指定使用特定GPU

    Args:
        gpu_idx (int): GPU索引
    """
    if isinstance(gpu_idx, int):
        gpu_idx = [gpu_idx]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_idx)
