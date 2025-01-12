import numpy as np
import librosa
import scipy.signal
from typing import Tuple
import matplotlib.pyplot as plt

# from fft import fft
from numpy.fft import fft


def create_window(window_length: int, window_type: str = "hann") -> np.ndarray:
    """
    Create a window function of specified type and length.

    Args:
        window_length (int): Length of the window in samples
        window_type (str): Type of window ('hann', 'hamming', 'blackman')

    Returns:
        np.ndarray: Window function
    """
    if window_type == "hann":
        return np.hanning(window_length)
    elif window_type == "hamming":
        return np.hamming(window_length)
    elif window_type == "blackman":
        return np.blackman(window_length)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")


def get_num_frames(signal_length: int, window_length: int, hop_length: int) -> int:
    """
    Calculate number of frames for given signal length.

    Args:
        signal_length (int): Length of input signal
        window_length (int): Length of the analysis window
        hop_length (int): Number of samples between frames

    Returns:
        int: Number of frames
    """
    # print(f"get num frames: {1 + (signal_length - window_length) // hop_length}")
    return 1 + (signal_length - window_length) // hop_length


def frame_signal(
    signal: np.ndarray,
    window_length: int,
    hop_length: int,
    window: np.ndarray,
) -> np.ndarray:
    """
    Segment signal into overlapping frames.

    Args:
        signal (np.ndarray): Input time domain signal
        window_length (int): Length of the analysis window
        hop_length (int): Number of samples between frames
        window (np.ndarray): Window function to apply to each frame

    Returns:
        np.ndarray: Matrix of frames (num_frames × window_length)
    """
    # print(f"raw signal: {signal[:10]}, ..., {signal[-10:]}")
    # todo 还有一个问题，这里其实是n_fft//2，而不是window_length//2
    signal = np.pad(signal, window_length // 2, mode="constant", constant_values=0)
    # signal = np.pad(signal, window_length // 2, mode="reflect")
    # print(
    #     f"padded signal: {signal[n_fft//2 - 5:n_fft//2+5]}, ..., {signal[-n_fft//2 - 5:-n_fft//2+5]}"
    # )
    num_frames = get_num_frames(len(signal), window_length, hop_length)
    frames = np.zeros((num_frames, window_length))

    for i in range(num_frames):
        start = i * hop_length
        end = start + window_length
        frames[i] = signal[start:end] * window

    return frames


def stand_stft(
    signal: np.ndarray,
    window_length: int = 2048,
    hop_length: int = 512,
    window_type: str = "hann",
):
    return (
        librosa.stft(
            signal, win_length=window_length, hop_length=hop_length, window=window_type
        ),
        None,
        None,
    )


def compute_stft(
    signal: np.ndarray,
    window_length: int = 2048,
    hop_length: int = 512,
    window_type: str = "hann",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Short-Time Fourier Transform of a signal.

    Args:
        signal (np.ndarray): Input time domain signal
        window_length (int): Length of the analysis window
        hop_length (int): Number of samples between frames
        window_type (str): Type of window function

    Returns:
        Tuple containing:
        - np.ndarray: STFT matrix (complex) [num_frames * (window_length//2+1)]
        - np.ndarray: Time points (frame centers in samples)
        - np.ndarray: Frequency points
    """
    # Create window function
    window = create_window(window_length, window_type)

    # Frame the signal
    frames = frame_signal(signal, window_length, hop_length, window)

    # Compute FFT for each frame
    stft_matrix = np.zeros((frames.shape[0], window_length), dtype=complex)
    for i in range(frames.shape[0]):
        stft_matrix[i] = fft(frames[i])

    # Calculate time and frequency points
    time_points = np.arange(frames.shape[0]) * hop_length
    freq_points = np.fft.fftfreq(window_length) * window_length
    # print(stft_matrix.shape)
    # print(stft_matrix[:, : window_length // 2 + 1].shape)
    return stft_matrix[:, : window_length // 2 + 1], time_points, freq_points


def main():
    # 设置全局字体
    # plt.rcParams["font.family"] = "Helvetica"

    # 生成测试信号
    duration = 5.0  # 秒
    signal, sample_rate = librosa.load("data/1-137-A-32.wav")

    # 计算STFT
    window_length = 256
    hop_length = 128
    stft_matrix, frequencies, times = compute_stft(signal, window_length, hop_length)

    # 只取正频率部分用于显示
    positive_frequencies_idx = frequencies >= 0
    spectrogram = np.abs(stft_matrix[positive_frequencies_idx, :])
    frequencies = frequencies[positive_frequencies_idx]

    # 转换为分贝单位
    spectrogram_db = 20 * np.log10(spectrogram + 1e-10)

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 绘制原始信号
    plt.subplot(211)
    time_axis = np.linspace(0, duration, len(signal))
    plt.plot(time_axis, signal)
    plt.title("Original Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Plot the spectrogram
    plt.subplot(212)
    plt.pcolormesh(times / sample_rate, frequencies, spectrogram_db, shading="gouraud")
    plt.title("STFT Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Magnitude (dB)")

    plt.tight_layout()
    plt.savefig("stft.png")


if __name__ == "__main__":
    main()
