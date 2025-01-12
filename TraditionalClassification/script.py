import subprocess
import itertools


def run_experiments():
    # 定义参数列表
    models = ["cnn", "crnn", "attention_cnn", "lstm", "resnet"]
    window_lengths = ["1024", "2048", "4096"]

    # 定义不同方法的配置
    configurations = [
        {"method": "raw", "batch_size": "128"},
        {"method": "fft", "batch_size": "200"},
        {"method": "mel", "batch_size": "1500"},
        {"method": "mfcc", "batch_size": "100", "transform": "dst"},
        {"method": "mfcc", "batch_size": "100", "transform": "dct"},
        {"method": "mfcc_derivatives", "batch_size": "100", "transform": "dst"},
        {"method": "mfcc_derivatives", "batch_size": "100", "transform": "dct"},
        {"method": "stft", "batch_size": "100"},
        {"method": "stft_DCT", "batch_size": "100"},
        {"method": "stft_derivatives", "batch_size": "100"},
    ]

    # 遍历所有组合
    for config in configurations:
        for model, window_length in itertools.product(models, window_lengths):
            print(f"Running experiment with model: {model}, window_length: {window_length}")

            # 构建命令
            cmd = ["python", "main.py"]
            cmd.extend(["--method", config["method"]])
            cmd.extend(["--batch_size", config["batch_size"]])
            cmd.extend(["--window_length", window_length])
            cmd.extend(["--model", model])

            # 添加transform参数（如果存在）
            if "transform" in config:
                cmd.extend(["--transform", config["transform"]])

            # 将命令转换为字符串（用于打印）
            cmd_str = " ".join(cmd)
            print(f"Executing: {cmd_str}")

            try:
                # 执行命令
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing command: {cmd_str}")
                print(f"Error: {str(e)}")
                continue


if __name__ == "__main__":
    run_experiments()
