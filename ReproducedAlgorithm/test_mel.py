import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import Tuple

from method.mfcc import compute_mel_spectrogram


def validate_mel_spectrogram(
    signal: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    window_length: int = 2048,
    hop_length: int = 512,
    f_min: float = 0.0,
    f_max: float = None,
) -> Tuple[float, dict]:
    """
    使用librosa的实现来验证自定义的mel谱计算函数。

    Args:
        signal: 输入音频信号
        sample_rate: 采样率
        n_mels: Mel滤波器组数量
        window_length: STFT窗口长度
        hop_length: STFT帧移
        f_min: 最小频率
        f_max: 最大频率

    Returns:
        Tuple包含:
        - 相关系数
        - 包含详细比较信息的字典
    """
    # 计算我们的实现结果
    mel_spec_ours, times_ours, freqs_ours = compute_mel_spectrogram(
        signal=signal,
        sample_rate=sample_rate,
        n_mels=n_mels,
        window_length=window_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
    )
    # # 不是power_to_db函数的差异
    # mel_spec_ours = librosa.power_to_db(mel_spec_ours)
    mel_spec_librosa = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=window_length,
        hop_length=hop_length,
        fmin=f_min,
        fmax=f_max,
        htk=True,
    )

    # 计算librosa的实现结果
    mel_spec_librosa_db = librosa.power_to_db(mel_spec_librosa)
    mel_spec_librosa_db = mel_spec_librosa_db.T  # 转置以匹配我们的输出格式

    # 计算相关系数
    correlation = np.corrcoef(mel_spec_ours.flatten(), mel_spec_librosa_db.flatten())[
        0, 1
    ]

    # 计算统计信息
    stats = {
        "mean_diff": np.mean(np.abs(mel_spec_ours - mel_spec_librosa_db)),
        "max_diff": np.max(np.abs(mel_spec_ours - mel_spec_librosa_db)),
        "std_diff": np.std(mel_spec_ours - mel_spec_librosa_db),
        "shape_match": mel_spec_ours.shape == mel_spec_librosa_db.shape,
        "our_shape": mel_spec_ours.shape,
        "librosa_shape": mel_spec_librosa_db.shape,
        "our_range": (np.min(mel_spec_ours), np.max(mel_spec_ours)),
        "librosa_range": (np.min(mel_spec_librosa_db), np.max(mel_spec_librosa_db)),
    }

    # 绘制比较图
    plt.figure(figsize=(15, 10))

    # 绘制我们的实现结果
    plt.subplot(2, 1, 1)
    plt.imshow(mel_spec_ours.T, aspect="auto", origin="lower")
    plt.colorbar(label="dB")
    plt.title("Our Implementation")
    plt.xlabel("Time Frame")
    plt.ylabel("Mel Band")

    # 绘制librosa的实现结果
    plt.subplot(2, 1, 2)
    plt.imshow(mel_spec_librosa_db.T, aspect="auto", origin="lower")
    plt.colorbar(label="dB")
    plt.title(f"Librosa Implementation (Correlation: {correlation:.4f})")
    plt.xlabel("Time Frame")
    plt.ylabel("Mel Band")

    plt.tight_layout()
    plt.savefig("image/mel_spectrogram_validation.png")
    plt.close()

    return correlation, stats


def print_validation_results(correlation: float, stats: dict) -> None:
    """
    打印验证结果。
    """
    print("\nMel Spectrogram Validation Results:")
    print("-" * 50)
    print(f"Our shape: {stats['our_shape']}")
    print(f"Librosa shape: {stats['librosa_shape']}")
    print(
        f"Our value range: [{stats['our_range'][0]:.2f}, {stats['our_range'][1]:.2f}]"
    )
    print(
        f"Librosa value range: [{stats['librosa_range'][0]:.2f}, {stats['librosa_range'][1]:.2f}]"
    )
    print(f"Correlation with librosa: {correlation:.4f}")
    print(f"Mean absolute difference: {stats['mean_diff']:.4f}")
    print(f"Maximum absolute difference: {stats['max_diff']:.4f}")
    print(f"Standard deviation of difference: {stats['std_diff']:.4f}")
    # print(f"Shape match: {stats['shape_match']}")


# 示例使用
if __name__ == "__main__":
    # 生成测试信号
    signal, sample_rate = librosa.load("data/1-137-A-32.wav")

    # 运行验证
    correlation, stats = validate_mel_spectrogram(signal, sample_rate, n_mels=40)

    # 打印结果
    print_validation_results(correlation, stats)
