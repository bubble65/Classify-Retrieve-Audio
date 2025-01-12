import librosa
import numpy as np
import matplotlib.pyplot as plt
from method.stft import compute_stft


# 2. 创建示例信号：一个随时间变化的正弦波
def create_test_signal(
    duration_seconds: float = 10.0, sample_rate: int = 44_100
) -> np.ndarray:
    """创建测试信号：频率从10Hz变化到50Hz的正弦波"""
    t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate))
    # 使用线性调频信号（Chirp信号）
    freq_start, freq_end = 10, 50000  # Hz
    phase = (
        2
        * np.pi
        * (freq_start * t + (freq_end - freq_start) * t**2 / (2 * duration_seconds))
    )
    return np.sin(phase)


def compare_stft_implementations(signal, sr, window_length=2048, hop_length=512):
    """
    比较不同的STFT实现结果

    Args:
        signal: 输入信号
        sr: 采样率
        window_length: 窗口长度
        hop_length: 帧移
    """
    our_stft, our_times, our_freqs = compute_stft(
        signal, window_length=window_length, hop_length=hop_length
    )

    # 2. 使用librosa的实现
    # librosa的STFT返回结果形状为(1 + n_fft//2, n_frames)，需要转置以匹配我们的实现
    librosa_stft = librosa.stft(
        signal,
        n_fft=window_length,
        hop_length=hop_length,
        win_length=window_length,
        window="hann",
        center=True,  # 这个参数控制是否在信号两端进行填充
    ).T

    # 3. 使用scipy的实现
    # scipy的实现需要明确指定nperseg（窗口长度）和noverlap（重叠样本数）
    # frequencies, times, scipy_stft = scipy.signal.stft(
    #     signal,
    #     fs=sr,
    #     nperseg=window_length,
    #     noverlap=window_length - hop_length,
    #     window="hann",
    # )
    # # 转置使形状匹配
    # scipy_stft = scipy_stft.T

    # 打印形状比较
    print("STFT 形状比较:")
    print(f"我们的实现: {our_stft.shape}")
    print(f"Librosa实现: {librosa_stft.shape}")
    # print(f"Scipy实现: {scipy_stft.shape}")

    # 计算幅度谱的相关系数
    corr_librosa = np.corrcoef(
        np.abs(our_stft).flatten(), np.abs(librosa_stft).flatten()
    )[0, 1]

    # corr_scipy = np.corrcoef(
    #     np.abs(our_stft_positive).flatten(), np.abs(scipy_stft).flatten()
    # )[0, 1]

    print("\n幅度谱相关系数:")
    print(f"与Librosa的相关系数: {corr_librosa:.4f}")

    mean_error = np.mean(np.abs(our_stft - librosa_stft))
    sum_error = np.sum(np.abs(our_stft - librosa_stft))
    print("mean error:", mean_error)
    print("sum error:", sum_error)

    # print(f"与Scipy的相关系数: {corr_scipy:.4f}")

    # 可视化比较
    plt.figure(figsize=(15, 10))

    # 1. 我们的实现
    plt.subplot(311)
    plt.title("our stft")
    plt.imshow(np.abs(our_stft).T, aspect="auto", origin="lower")
    plt.colorbar(label="range")

    # 2. Librosa的实现
    plt.subplot(312)
    plt.title("Librosa")
    plt.imshow(np.abs(librosa_stft).T, aspect="auto", origin="lower")
    plt.colorbar(label="range")

    print(our_stft)
    print("\n\n\n")
    print(librosa_stft)
    # todo 这里的问题应该并不是特别大

    # print(f"{(np.abs(our_stft_positive) - np.abs(librosa_stft))}")

    # 3. Scipy的实现
    # plt.subplot(313)
    # plt.title("Scipy的STFT实现")
    # plt.imshow(np.abs(scipy_stft).T, aspect="auto", origin="lower")
    # plt.colorbar(label="幅度")

    plt.tight_layout()
    plt.savefig("image/stft_compare.png")

    return our_stft, librosa_stft, None


# 创建测试信号并运行比较
def run_test():
    signal, sr = librosa.load("data/1-137-A-32.wav")

    # 运行比较
    our_stft, librosa_stft, scipy_stft = compare_stft_implementations(signal, sr)

    return our_stft, librosa_stft, scipy_stft


def main():
    run_test()


if __name__ == "__main__":
    main()
