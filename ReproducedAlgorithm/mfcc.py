import numpy as np
from typing import Tuple
from method.stft import compute_stft
import matplotlib.pyplot as plt
import librosa


def hz_to_mel(hz: float) -> float:
    """Convert frequency from Hz to Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    """Convert frequency from Mel scale to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    n_fft: int,
    sample_rate: int,
    n_mels: int = 40,
    f_min: float = 0.0,
    f_max: float = None,
) -> np.ndarray:
    """
    Create a Mel filterbank matrix.

    Args:
        n_fft (int): FFT size
        sample_rate (int): Audio sample rate in Hz
        n_mels (int): Number of Mel bands
        f_min (float): Minimum frequency for Mel filterbank
        f_max (float): Maximum frequency for Mel filterbank

    Returns:
        np.ndarray: Mel filterbank matrix (n_mels, (n_fft//2 + 1))
    """
    f_max = f_max or sample_rate / 2.0

    # Convert frequency range to Mel scale
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    # Create equally spaced points in Mel scale
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz(mel) for mel in mel_points])  # 对应的mel_f

    # Convert Hz points to FFT bin numbers
    # bin_numbers = (n_fft + 1) * hz_points / sample_rate
    # 不使用bin_numbers，而是使用ramps

    fft_freqs = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))

    for i in range(n_mels):
        # 计算三角滤波器的左右边界
        lower = hz_points[i]
        peak = hz_points[i + 1]
        upper = hz_points[i + 2]

        up_slope = (fft_freqs - lower) / (peak - lower)
        down_slope = (upper - fft_freqs) / (upper - peak)
        filterbank[i] = np.maximum(0, np.minimum(up_slope, down_slope))

    # Create filterbank matrix
    # filterbank = np.zeros((n_mels, n_fft // 2 + 1))

    # for i in range(n_mels):
    #     for j in range(int(bin_numbers[i]), int(bin_numbers[i + 1])):
    #         filterbank[i, j] = (j - bin_numbers[i]) / (
    #             bin_numbers[i + 1] - bin_numbers[i]
    #         )
    #     for j in range(int(bin_numbers[i + 1]), int(bin_numbers[i + 2])):
    #         filterbank[i, j] = (bin_numbers[i + 2] - j) / (
    #             bin_numbers[i + 2] - bin_numbers[i + 1]
    #         )
    # enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    # filterbank *= enorm[:, np.newaxis]
    filterbank = (
        filterbank
        * 2.0
        / (hz_points[2 : n_mels + 2] - hz_points[:n_mels]).reshape(-1, 1)
    )
    return filterbank


def stand_mfcc(
    signal: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    n_mels: int = 40,
    window_length: int = 2048,
    hop_length: int = 512,
    f_min: float = 0.0,
    f_max: float = None,
    window_type: str = "hann",
):
    return (
        librosa.feature.mfcc(
            signal,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            n_fft=window_length,
            hop_length=hop_length,
            fmin=f_min,
            fmax=f_max,
            window=window_type,
            htk=True,
        ).transpose(),
        None,
        None,
    )


def create_linear_filterbank(
    n_fft: int,
    sample_rate: int,
    n_bands: int = 40,
    f_min: float = 0.0,
    f_max: float = None,
) -> np.ndarray:
    """
    Create a linear filterbank matrix with overlapping triangular filters.

    Args:
        n_fft: FFT size
        sample_rate: Audio sample rate in Hz
        n_bands: Number of frequency bands
        f_min: Minimum frequency
        f_max: Maximum frequency

    Returns:
        np.ndarray: Filterbank matrix (n_bands, n_fft//2 + 1)
    """
    f_max = f_max or sample_rate / 2.0

    # Create linearly spaced frequency points
    freq_points = np.linspace(f_min, f_max, n_bands + 2)

    # Convert frequencies to FFT bin numbers
    bins = np.floor((n_fft + 1) * freq_points / sample_rate)

    # Create filterbank matrix
    filterbank = np.zeros((n_bands, n_fft // 2 + 1))

    # Create overlapping triangular filters
    for i in range(n_bands):
        # Left side of triangle
        for j in range(int(bins[i]), int(bins[i + 1])):
            filterbank[i, j] = (j - bins[i]) / (bins[i + 1] - bins[i])
        # Right side of triangle
        for j in range(int(bins[i + 1]), int(bins[i + 2])):
            filterbank[i, j] = (bins[i + 2] - j) / (bins[i + 2] - bins[i + 1])

    return filterbank


def stand_mel(
    signal: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    window_length: int = 2048,
    hop_length: int = 512,
    f_min: float = 0.0,
    f_max: float = None,
    power: float = 2.0,
    db_scale: bool = True,
    ref: float = 1.0,
    top_db: float = 80.0,
):
    """
    db_scale (bool): Whether to convert to decibels
    ref (float): Reference value for db conversion
    top_db (float): Threshold the output to top_db below peak
    """
    return (
        librosa.feature.melspectrogram(
            y=signal,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=window_length,
            hop_length=hop_length,
            fmin=f_min,
            fmax=f_max,
            power=power,
        ),
        None,
        None,
    )


def compute_mel_spectrogram(
    signal: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    window_length: int = 2048,
    hop_length: int = 512,
    f_min: float = 0.0,
    f_max: float = None,
    power: float = 2.0,
    db_scale: bool = True,
    window_type: str = "hann",
    ref: float = 1.0,
    top_db: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Mel spectrogram of an audio signal.

    Args:
        signal (np.ndarray): Input audio signal
        sample_rate (int): Audio sample rate in Hz
        n_mels (int): Number of Mel bands
        window_length (int): STFT window length
        hop_length (int): STFT hop length
        f_min (float): Minimum frequency for Mel filterbank
        f_max (float): Maximum frequency for Mel filterbank
        power (float): Exponent for the magnitude spectrogram
        db_scale (bool): Whether to convert to decibels
        (no use) ref (float): Reference value for db conversion
        (no use) top_db (float): Threshold the output to top_db below peak

    Returns:
        Tuple containing:
        - np.ndarray: Mel spectrogram matrix, shape = (num_frames, n_mels)
        - np.ndarray: Time points
        - np.ndarray: Mel frequency points
    """

    stft_matrix, time_points, _ = compute_stft(
        signal, window_length, hop_length, window_type=window_type
    )

    # Create Mel filterbank
    mel_filterbank = create_mel_filterbank(
        window_length, sample_rate, n_mels, f_min, f_max
    )
    # mel_filterbank = librosa.filters.mel(
    #     sr=sample_rate,
    #     n_fft=window_length,
    #     n_mels=n_mels,
    #     fmin=f_min,
    #     fmax=f_max,
    #     htk=True,
    # )

    power_spectrum = np.abs(stft_matrix) ** power

    # Apply Mel filterbank
    # mel_spectrum = np.dot(power_spectrum, mel_filterbank.T)
    mel_spectrum = np.einsum(
        "...ft,mf->...mt", power_spectrum.T, mel_filterbank, optimize=True
    )
    mel_spectrum = mel_spectrum.T
    # Convert to decibels if requested
    if db_scale:
        mel_spectrum = power_to_db(mel_spectrum)

    # Calculate Mel frequency points for visualization
    if f_max is None:
        f_max = sample_rate / 2.0

    mel_freqs = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels)
    mel_freqs = np.array([mel_to_hz(mel) for mel in mel_freqs])
    # print("mel_shape:", mel_spectrum.shape)

    return mel_spectrum, time_points, mel_freqs


def compute_stft_DCT(
    signal: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    n_mels: int = 40,
    window_length: int = 2048,
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
        Tuple containing:
        - np.ndarray: Feature matrix. Shape = (n_mfcc, num_frames)
        - np.ndarray: Time points
        - np.ndarray: Coefficient indices
    """
    stft_matrix, time_points, _ = compute_stft(
        signal,
        window_length=window_length,
        hop_length=hop_length,
        window_type=window_type,
    )
    # stft_matrix = stft_matrix[:, : window_length // 2 + 1]
    power_spectrum = np.abs(stft_matrix) ** 2
    # Create linearly spaced frequency bands instead of Mel bands
    # We'll average the power spectrum into these bands
    # n_freqs = power_spectrum.shape[1]
    # bands = np.linspace(0, n_freqs, n_mels + 1).astype(int)
    # 创建线性filterbank
    filterbank = create_linear_filterbank(window_length, sample_rate, n_bands=n_mels)

    # 应用filterbank
    filtered_spectrum = np.einsum(
        "...ft,mf->...mt", power_spectrum.T, filterbank, optimize=True
    ).T

    # Average the power spectrum into linear frequency bands
    # linear_spectrum = np.zeros((power_spectrum.shape[0], n_mels))
    # for i in range(n_mels):
    #     linear_spectrum[:, i] = np.mean(
    #         power_spectrum[:, bands[i] : bands[i + 1]], axis=1
    #     )

    # Take log of spectrum
    # spectrum = np.log(filtered_spectrum + 1e-10)
    spectrum = power_to_db(filtered_spectrum)

    # Apply DCT
    features = np.zeros((spectrum.shape[0], n_mfcc))
    if transform.lower() == "dct":
        for i in range(n_mfcc):
            features[:, i] = np.sum(
                spectrum * np.cos(np.pi * i / n_mels * (np.arange(n_mels) + 0.5)),
                axis=1,
            )
    elif transform.lower() == "dst":
        for i in range(n_mfcc):
            features[:, i] = np.sum(
                spectrum * np.sin(np.pi * i / n_mels * (np.arange(n_mels) + 0.5)),
                axis=1,
            )
    features = features.T

    # Calculate time points
    time_points = np.arange(len(features)) * hop_length / sample_rate

    return features, time_points, np.arange(n_mfcc)


def compute_mfcc(
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    Returns:
        Tuple containing:
        - np.ndarray: MFCC matrix. Shape = (n_mfcc, num_frames)
        - np.ndarray: Time points
        - np.ndarray: MFCC coefficient indices
    """
    mel_spectrum, time_points, _ = compute_mel_spectrogram(
        signal,
        sample_rate,
        n_mels=n_mels,
        window_length=window_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        window_type=window_type,
    )

    # Take log of Mel spectrum
    # mel_spectrum = np.log(mel_spectrum + 1e-10)
    # print(mel_spectrum.shape)

    # Apply DCT to get MFCCs
    # mfcc = scipy.fftpack.dct(S, axis=-2, type=2, norm="ortho")[..., :n_mfcc, :]
    if transform == "dct":
        mfcc = np.zeros((mel_spectrum.shape[0], n_mfcc))
        for i in range(n_mfcc):
            mfcc[:, i] = np.sum(
                mel_spectrum * np.cos(np.pi * i / n_mels * (np.arange(n_mels) + 0.5)),
                axis=1,
            )
    elif transform == "dst":
        mfcc = np.zeros((mel_spectrum.shape[0], n_mfcc))
        for i in range(n_mfcc):
            mfcc[:, i] = np.sum(
                mel_spectrum * np.sin(np.pi * i / n_mels * (np.arange(n_mels) + 0.5)),
                axis=1,
            )
    return mfcc.T, time_points, np.arange(n_mfcc)


def compute_deltas(features: np.ndarray) -> np.ndarray:
    """
    Compute delta features (first derivative).

    Args:
        features (np.ndarray): Input feature matrix

    Returns:
        np.ndarray: Delta features
    """
    return np.gradient(features, axis=1)


def compute_mfcc_with_derivatives(
    signal: np.ndarray, sample_rate: int, gradient=1, **kwargs
) -> np.ndarray:
    """
    Compute MFCCs along with their delta and delta-delta features.

    Args:
        signal (np.ndarray): Input audio signal
        sample_rate (int): Audio sample rate in Hz
        **kwargs: Additional arguments to pass to compute_mfcc()

    Returns:
        np.ndarray: Matrix containing MFCCs and their derivatives, shape = (n_mfcc*(1+gradient), num_frames)

    """
    mfcc, _, _ = compute_mfcc(signal, sample_rate, **kwargs)

    # Compute deltas
    delta = np.gradient(mfcc, axis=1)
    delta2 = np.gradient(delta, axis=1)
    if gradient == 1:
        return np.vstack([mfcc, delta])
    return np.vstack([mfcc, delta, delta2])


def compute_wavelet_transform(
    signal: np.ndarray,
    sample_rate: int,
    wavelet: str = "cmor1.5-1.0",
    num_scales: int = 128,
    freq_min: float = 20,
    freq_max: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute continuous wavelet transform of the signal.

    Args:
        signal: Input audio signal
        sample_rate: Sampling rate in Hz
        wavelet: Wavelet type to use
        num_scales: Number of scales for wavelet transform
        freq_min: Minimum frequency
        freq_max: Maximum frequency

    Returns:
        Tuple of (wavelet coefficients, frequencies)
    """
    if freq_max is None:
        freq_max = sample_rate / 2

    # Calculate scales for desired frequency range
    freq = np.linspace(freq_min, freq_max, num_scales)
    scales = pywt.frequency2scale(wavelet, freq, sample_rate)

    # Compute CWT
    coef, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1 / sample_rate)

    return coef, freq


def compute_wavelet_mfcc(
    signal: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    n_mels: int = 40,
    wavelet: str = "cmor1.5-1.0",
    num_scales: int = 128,
    f_min: float = 20.0,
    f_max: float = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute MFCC using wavelet transform instead of STFT.

    Args:
        signal: Input audio signal
        sample_rate: Audio sample rate in Hz
        n_mfcc: Number of MFCC coefficients to return
        n_mels: Number of Mel bands
        wavelet: Wavelet type to use
        num_scales: Number of scales for wavelet transform
        f_min: Minimum frequency
        f_max: Maximum frequency

    Returns:
        Tuple containing:
        - MFCC matrix (n_mfcc × num_frames)
        - Time points
        - MFCC coefficient indices
    """
    # Compute wavelet transform
    wavelet_coef, frequencies = compute_wavelet_transform(
        signal, sample_rate, wavelet, num_scales, f_min, f_max
    )

    # Convert to power spectrum
    power_spectrum = np.abs(wavelet_coef) ** 2

    # Create and apply Mel filterbank
    mel_filterbank = create_mel_filterbank(
        power_spectrum.shape[0], sample_rate, n_mels, f_min, f_max
    )

    # Apply Mel filterbank
    mel_spectrum = np.dot(mel_filterbank, power_spectrum)

    # Take log of Mel spectrum
    mel_spectrum = np.log(mel_spectrum + 1e-10)

    # Apply DCT
    mfcc = np.zeros((n_mfcc, mel_spectrum.shape[1]))
    for i in range(n_mfcc):
        mfcc[i] = np.sum(
            mel_spectrum
            * np.cos(np.pi * i / n_mels * (np.arange(n_mels) + 0.5))[:, np.newaxis],
            axis=0,
        )

    # Calculate time points
    time_points = np.arange(mfcc.shape[1]) / sample_rate

    return mfcc, time_points, np.arange(n_mfcc)


def compute_wavelet_mfcc_with_derivatives(
    signal: np.ndarray, sample_rate: int, **kwargs
) -> np.ndarray:
    """
    Compute wavelet-based MFCCs with delta and delta-delta features.

    Args:
        signal: Input audio signal
        sample_rate: Audio sample rate in Hz
        **kwargs: Additional arguments to pass to compute_wavelet_mfcc()

    Returns:
        Matrix containing MFCCs and their derivatives
    """
    mfcc, _, _ = compute_wavelet_mfcc(signal, sample_rate, **kwargs)

    # Compute deltas
    delta = np.gradient(mfcc, axis=1)
    delta2 = np.gradient(delta, axis=1)

    # Concatenate features
    return np.vstack([mfcc, delta, delta2])


def power_to_db(S: np.ndarray, ref: float = 1.0, top_db: float = 80.0) -> np.ndarray:
    """
    Convert a power spectrogram to decibel units.

    Args:
        S (np.ndarray): Input power spectrogram
        ref (float): Reference value for db conversion
        top_db (float): Threshold the output to top_db below peak

    Returns:
        np.ndarray: Power spectrogram in dB units
    """
    # Convert to decibels
    magnitude_db = 10.0 * np.log10(np.maximum(1e-10, S))
    magnitude_db -= 10.0 * np.log10(np.maximum(1e-10, ref))

    # Threshold output
    if top_db is not None:
        magnitude_db = np.maximum(magnitude_db, magnitude_db.max() - top_db)

    return magnitude_db


def plot_mel_spectrogram(
    signal: np.ndarray, sample_rate: int, title: str = "Mel Spectrogram"
) -> None:
    """
    Plot the Mel spectrogram of an audio signal.

    Args:
        signal (np.ndarray): Input audio signal
        sample_rate (int): Audio sample rate in Hz
        title (str): Plot title
    """
    # Compute Mel spectrogram
    mel_spec, times, mel_freqs = compute_mel_spectrogram(signal, sample_rate)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot Mel spectrogram
    plt.pcolormesh(times / sample_rate, mel_freqs, mel_spec.T, shading="gouraud")

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Magnitude (dB)")

    # Use log scale for frequency axis
    plt.yscale("log")
    plt.ylim(mel_freqs[0], mel_freqs[-1])

    plt.tight_layout()


def main():
    """
    Example usage and visualization of Mel spectrogram.
    """
    # Generate a test signal with varying frequency components
    signal, sample_rate = librosa.load("data/1-137-A-32.wav")
    compute_mel_spectrogram(signal, sample_rate)
    # Plot the Mel spectrogram
    # plot_mel_spectrogram(signal, sample_rate, "Mel Spectrogram of Chirp Signal")
    # plt.savefig("image/mel_spectrogram.png")
    # plt.close()


if __name__ == "__main__":
    main()
