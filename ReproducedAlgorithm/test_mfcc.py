import numpy as np
from method.mfcc import (
    compute_mfcc,
    create_mel_filterbank,
    compute_mfcc_with_derivatives,
)
import librosa
import matplotlib.pyplot as plt


def generate_test_signal(duration=1.0, sample_rate=16000):
    """
    Generate a test signal containing multiple frequency components.
    Creates a more complex signal than a simple sine wave to better demonstrate MFCC features.
    """
    t = np.linspace(0, duration, int(duration * sample_rate))

    # Create a signal with multiple frequency components
    signal = (
        np.sin(2 * np.pi * 440 * t)  # A4 note (440 Hz)
        + 0.5 * np.sin(2 * np.pi * 880 * t)  # First harmonic
        + 0.25 * np.sin(2 * np.pi * 1760 * t)  # Second harmonic
    )

    # Add some time-varying characteristics
    envelope = np.exp(-3 * t)  # Decay envelope
    signal *= envelope

    return signal


def compare_with_librosa(signal, sample_rate):
    """
    Compare our MFCC implementation with librosa's implementation.
    """

    # Compute MFCCs using our implementation
    n_mfcc = 13
    n_mel = 40
    window_length = 1024
    hop_length = 512
    our_mfccs, _, _ = compute_mfcc(
        signal,
        sample_rate,
        n_mfcc=n_mfcc,
        n_mels=n_mel,
        window_length=window_length,
        hop_length=hop_length,
    )
    print("Our implementation:", our_mfccs.shape)

    librosa_mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_mels=n_mel,
        n_fft=window_length,
        hop_length=hop_length,
        pad_mode="constant",
        htk=True,
    )
    print("librosa_mfccs implementation:", librosa_mfccs.shape)
    # librosa_mfccs = librosa_mfccs.transpose()

    correlations = [
        np.corrcoef(our_mfccs[i, :], librosa_mfccs[i, :])[0, 1] for i in range(13)
    ]

    # Compare filterbanks
    our_filter_bank = create_mel_filterbank(window_length, sample_rate, n_mel)
    print("our_filter_bank implementation:", our_filter_bank.shape)

    librosa_filter_bank = librosa.filters.mel(
        sr=sample_rate, n_fft=window_length, n_mels=n_mel, htk=True
    )
    print("librosa_filter_bank implementation:", librosa_filter_bank.shape)

    mean_error = np.mean(np.abs(our_filter_bank - librosa_filter_bank))
    sum_error = np.sum(np.abs(our_filter_bank - librosa_filter_bank))
    print("filter_bank mean error:", mean_error)
    print("filter_bank sum error:", sum_error)
    print(our_filter_bank)
    print("===")
    print(librosa_filter_bank)
    return correlations or [1]


def plot_mfcc_analysis(signal, sample_rate, title="MFCC Analysis"):
    """
    Perform MFCC analysis and create visualizations of the results.
    """
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    fig.suptitle(title, fontsize=16)

    # Plot 1: Original Signal
    axes[0].plot(np.arange(len(signal)) / sample_rate, signal)
    axes[0].set_title("Original Signal")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    # Compute and plot MFCCs
    mfccs, times, mfcc_indices = compute_mfcc(
        signal, sample_rate, n_mfcc=13, n_mels=40, window_length=2048, hop_length=512
    )

    # Plot 2: MFCC Coefficients
    im = axes[1].imshow(
        mfccs.T,
        aspect="auto",
        origin="lower",
        extent=[0, len(signal) / sample_rate, 0, mfccs.shape[1]],
    )
    axes[1].set_title("MFCC Coefficients")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("MFCC Coefficient")
    plt.colorbar(im, ax=axes[1], label="Magnitude")

    # Compute and plot MFCCs with derivatives
    mfcc_with_deltas = compute_mfcc_with_derivatives(signal, sample_rate, n_mfcc=13, gradient=2)
    print(mfcc_with_deltas.shape)

    # Plot 3: MFCCs with Derivatives
    im = axes[2].imshow(
        mfcc_with_deltas.T,
        aspect="auto",
        origin="lower",
        extent=[0, len(signal) / sample_rate, 0, mfcc_with_deltas.shape[1]],
    )
    axes[2].set_title("MFCCs with Delta and Delta-Delta")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Feature Index")
    plt.colorbar(im, ax=axes[2], label="Magnitude")

    # Plot 4: First MFCC coefficient and its derivatives
    axes[3].plot(times / sample_rate, mfcc_with_deltas[0, :].T, label="MFCC[0]")
    axes[3].plot(times / sample_rate, mfcc_with_deltas[13, :].T, label="Delta[0]")
    axes[3].plot(times / sample_rate, mfcc_with_deltas[26, :].T, label="Delta-Delta[0]")
    axes[3].set_title("First MFCC Coefficient and its Derivatives")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Magnitude")
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    return fig


def main():
    """
    Main function to run tests and generate visualizations.
    """
    # Generate test signal
    sample_rate = 44_100
    signal = generate_test_signal(duration=5.0, sample_rate=sample_rate)

    signal, sample_rate = librosa.load("data/1-137-A-32.wav")

    # Create visualization
    fig = plot_mfcc_analysis(signal, sample_rate)
    plt.savefig("image/mfcc_analysis.png")

    # Compare with librosa
    correlations = compare_with_librosa(signal, sample_rate)
    # print(correlations)
    print("\nCorrelation with librosa implementation:")
    for i, corr in enumerate(correlations):
        print(f"MFCC {i}: {corr:.4f}")


if __name__ == "__main__":
    main()
