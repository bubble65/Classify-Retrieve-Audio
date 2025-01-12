import librosa
import numpy as np
import matplotlib.pyplot as plt

def extract_f0_using_cepstrum(y, sr, frame_length=2048, hop_length=512):
    """
    使用倒谱法提取基频（F0）

    Args:
        y (np.ndarray): 音频信号
        sr (int): 采样率
        frame_length (int): 帧长度
        hop_length (int): 帧移长度

    Returns:
        np.ndarray: 每帧的基频（F0）
    """
    # 预加重
    y_pre = librosa.effects.preemphasis(y)

    # 分帧
    frames = librosa.util.frame(y_pre, frame_length=frame_length, hop_length=hop_length)

    # 加窗
    window = np.hamming(frame_length)
    frames = frames * window[:, np.newaxis]

    # 计算频谱
    spectrum = np.fft.rfft(frames, axis=0)

    # 取对数幅度谱
    log_spectrum = np.log(np.abs(spectrum) + 1e-6)

    # 计算倒谱
    cepstrum = np.fft.irfft(log_spectrum, axis=0)
    cepstrum[0] = 0
    # 寻找倒谱中的峰值
    peaks = np.argmax(cepstrum[1:frame_length // 2], axis=0) + 1

    # 计算基频
    f0 = sr / peaks
    print(f0.shape)
    print(peaks.shape)
    print(cepstrum.shape)
    # # 提取某一帧的频谱
    # frame_index = 2 # 选择某一帧的索引
    # frame_spectrum = np.abs(spectrum[:, frame_index])  # 提取该帧的幅度谱

    # # 对幅度谱取对数 (Log Spectrum)
    # log_spectrum = np.log(frame_spectrum + 1e-6)  # 加一个小值避免 log(0)

    # # 计算倒谱 (Cepstrum)
    # cepstrum = np.fft.ifft(log_spectrum)

    # # 取实部 (因为 IFFT 可能包含虚部)
    # cepstrum_real = np.real(cepstrum)
    # cepstrum_real[0] = 0
    # # 计算 quefrency 轴 (倒谱的频率轴)
    # quefrency = np.arange(len(cepstrum_real)) / sr
    # # 绘制某一帧的倒谱图像
    # plt.figure(figsize=(10, 6))
    # plt.plot(quefrency, cepstrum_real, label='Cepstrum (倒谱)')
    # plt.title(f'Cepstrum of Frame {frame_index}')
    # plt.xlabel('Quefrency (s)')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    return f0
