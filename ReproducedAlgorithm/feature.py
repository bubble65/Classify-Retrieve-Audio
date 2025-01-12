"""
各种特征
- 统计特征，在函数`static_features`里面
- 魔改的stft
    - stft + 梯度 (指定gradient 参数)
    - stft + dct/dst
- 魔改的mfcc
    - mfcc + 梯度 (指定gradient 参数)
    - mfcc + dct/dst
"""

import numpy as np
import librosa
from typing import Dict
from scipy import stats
from .stft import compute_stft
from .mfcc import compute_stft_DCT, compute_mfcc


def static_features(signal: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """
    Extract statistical and audio-specific features from an audio signal.

    Args:
        signal (np.ndarray): Input audio signal
        sample_rate (int): Sampling rate of the audio

    Returns:
        Dict[str, float]: Dictionary containing various audio features
    """
    features = {}

    # Time domain features
    features["rms_energy"] = np.sqrt(np.mean(signal**2))
    features["zero_crossing_rate"] = np.mean(np.abs(np.diff(np.signbit(signal))))
    features["mean_amplitude"] = np.mean(np.abs(signal))
    features["std_amplitude"] = np.std(signal)
    features["max_amplitude"] = np.max(np.abs(signal))
    features["min_amplitude"] = np.min(np.abs(signal))

    # Statistical features
    features["skewness"] = stats.skew(signal)
    features["kurtosis"] = stats.kurtosis(signal)
    features["crest_factor"] = np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2))

    # Spectral features
    stft = np.abs(librosa.stft(signal))

    # Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)[0]
    features["spectral_centroid_mean"] = np.mean(spectral_centroids)
    features["spectral_centroid_std"] = np.std(spectral_centroids)

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate)[0]
    features["spectral_rolloff_mean"] = np.mean(rolloff)
    features["spectral_rolloff_std"] = np.std(rolloff)

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)[0]
    features["spectral_bandwidth_mean"] = np.mean(bandwidth)
    features["spectral_bandwidth_std"] = np.std(bandwidth)

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sample_rate)
    features["spectral_contrast_mean"] = np.mean(contrast)
    features["spectral_contrast_std"] = np.std(contrast)

    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(y=signal)[0]
    features["spectral_flatness_mean"] = np.mean(flatness)
    features["spectral_flatness_std"] = np.std(flatness)

    # Tempo and rhythmic features
    onset_env = librosa.onset.onset_strength(y=signal, sr=sample_rate)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)
    features["tempo"] = tempo[0]

    # Harmonic and percussive components
    harmonic, percussive = librosa.effects.hpss(signal)
    features["harmonic_mean"] = np.mean(np.abs(harmonic))
    features["percussive_mean"] = np.mean(np.abs(percussive))
    features["harmonic_percussive_ratio"] = features["harmonic_mean"] / (
        features["percussive_mean"] + 1e-10
    )

    # Root Mean Square Energy frames
    rms_frames = librosa.feature.rms(y=signal)[0]
    features["rms_mean"] = np.mean(rms_frames)
    features["rms_std"] = np.std(rms_frames)
    features["rms_max"] = np.max(rms_frames)

    return features


def stft_gradient(
    signal: np.ndarray,
    win_length: int = 2048,
    hop_length: int = 512,
    window_type: str = "hann",
    gradient: int = 1,
):
    """
    return stft + gradient:
        shape: (num_frames, (window_length//2+1) * gradient)
    """
    stft_matrix, _, _ = compute_stft(signal, win_length, hop_length, window_type)

    if gradient == 1:
        d = np.gradient(stft_matrix, axis=0)
        return np.hstack([stft_matrix, d])
    elif gradient == 2:
        d1 = np.gradient(stft_matrix, axis=0)
        d2 = np.gradient(d1, axis=0)
        return np.hstack([stft_matrix, d1, d2])


def stft_dct(
    signal: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    n_mels: int = 40,
    win_length: int = 2048,
    hop_length: int = 512,
    window_type: str = "hann",
    transform: str = "dct",
):
    """
    Compute audio features using STFT and DCT without Mel-scale filtering.

    Args:
        signal (np.ndarray): Input audio signal
        sample_rate (int): Audio sample rate in Hz
        n_mfcc (int): Number of DCT coefficients to return
        n_mels (int): Number of frequency bands for linear spacing
        window_length (int): STFT window length
        hop_length (int): STFT hop length
        window_type (str): Window type for STFT
        transform (str): Type of transform to apply (dct or dst)

    Returns:
        - np.ndarray: Feature matrix. Shape = (n_mfcc, num_frames)
    """
    return compute_stft_DCT(
        signal,
        sample_rate,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        window_length=win_length,
        hop_length=hop_length,
        window_type=window_type,
        transform=transform,
    )[0]


def mfcc_gradient(
    signal: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    n_mels: int = 40,
    window_length: int = 2048,
    hop_length: int = 512,
    f_min: float = 0.0,
    f_max: float = None,
    window_type: str = "hann",
    transform: str = "dct",
    gradient: int = 1,
):
    """
    Compute MFCCs for a given signal.

    Args:
        signal (np.ndarray): Input audio signal
        sample_rate (int): Audio sample rate in Hz
        n_mfcc (int): Number of MFCCs to return
        n_mels (int): Number of Mel bands
        window_length (int): STFT window length
        hop_length (int): STFT hop length
        f_min (float): Minimum frequency for Mel filterbank
        f_max (float): Maximum frequency for Mel filterbank
        window_type: Window type for STFT
        transform: Type of transform to apply (dct or dst)
        gradient: Number of gradient to compute

    Returns:
        - np.ndarray: MFCC matrix. Shape = (n_mfcc*(1+gradient), num_frames)
            - 把梯度作为一种特征来返回
    """
    mfcc_matrix, _, _ = compute_mfcc(
        signal,
        sample_rate,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        window_length=window_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        window_type=window_type,
        transform=transform,
    )

    if gradient == 1:
        d = np.gradient(mfcc_matrix, axis=1)
        return np.vstack([mfcc_matrix, d])
    elif gradient == 2:
        d1 = np.gradient(mfcc_matrix, axis=1)
        d2 = np.gradient(d1, axis=1)
        return np.vstack([mfcc_matrix, d1, d2])

    return mfcc_matrix
