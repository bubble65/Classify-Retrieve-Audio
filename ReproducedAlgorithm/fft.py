import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def custom_fft(a: np.ndarray, inverse: bool = False) -> np.ndarray:
    a = np.asarray(a, dtype=np.complex128)
    n = len(a)

    if n == 1:
        return a

    if n & (n - 1) != 0:
        raise ValueError("Length must be power of 2")

    even = custom_fft(a[::2], inverse)
    odd = custom_fft(a[1::2], inverse)

    factor = 1 if inverse else -1
    omega = np.exp(factor * 2j * np.pi * np.arange(n // 2) / n)
    result = np.zeros(n, dtype=np.complex128)

    result[: n // 2] = even + omega * odd
    result[n // 2 :] = even - omega * odd

    return result


def ifft(a: np.ndarray) -> np.ndarray:
    n = len(a)
    return np.array(custom_fft(a, inverse=True)) / n


def verify_fft_implementation(signal: np.ndarray) -> Tuple[float, float, float]:
    """
    Verify custom FFT implementation against numpy.fft.fft

    Args:
        signal (np.ndarray): Input signal to test

    Returns:
        Tuple[float, float, float]:
            - Relative error between FFT results
            - Maximum absolute difference
            - Correlation coefficient
    """
    # Ensure input length is power of 2
    n = len(signal)
    if n & (n - 1) != 0:
        raise ValueError("Input length must be a power of 2")

    # Compute FFT using both implementations
    custom_result = np.array(custom_fft(signal))
    numpy_result = np.fft.fft(signal)

    # Calculate error metrics
    abs_diff = np.abs(custom_result - numpy_result)
    max_diff = np.max(abs_diff)
    relative_error = np.mean(abs_diff / (np.abs(numpy_result) + 1e-10))
    correlation = np.corrcoef(np.abs(custom_result), np.abs(numpy_result))[0, 1]

    # Print detailed comparison for debugging
    print("\nDetailed comparison:")
    print("First few values comparison:")
    for i in range(min(5, len(custom_result))):
        print(f"Index {i}:")
        print(f"Custom: {custom_result[i]:.4f}")
        print(f"NumPy:  {numpy_result[i]:.4f}")
        print(f"Diff:   {abs_diff[i]:.4f}\n")

    return relative_error, max_diff, correlation


def test_fft():
    """
    Run comprehensive tests on the FFT implementation
    """
    # Test case 1: Simple sinusoid
    N = 16  # Using smaller size for better debugging
    t = np.linspace(0, 1, N)
    signal1 = np.sin(2 * np.pi * 2 * t)  # Lower frequency for smaller N

    # Test case 2: Sum of sinusoids
    signal2 = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 4 * t)

    # Test case 3: Complex signal
    signal3 = np.exp(2j * np.pi * 2 * t)

    # Test case 4: Random signal
    np.random.seed(42)
    signal4 = np.random.randn(N)

    test_signals = {
        "Single Sinusoid": signal1,
        "Multiple Sinusoids": signal2,
        "Complex Exponential": signal3,
        "Random Signal": signal4,
    }

    # Run tests and plot results
    plt.figure(figsize=(15, 10))

    for idx, (name, signal) in enumerate(test_signals.items(), 1):
        print(f"\n=== Testing {name} ===")
        rel_error, max_diff, corr = verify_fft_implementation(signal)

        # Plot original signal and FFT comparison
        plt.subplot(4, 2, 2 * idx - 1)
        plt.plot(np.real(signal))
        plt.title(f"{name} - Original Signal")
        plt.grid(True)

        custom_fft_result = np.array(custom_fft(signal))
        numpy_fft_result = np.fft.fft(signal)

        plt.subplot(4, 2, 2 * idx)
        plt.plot(np.abs(custom_fft_result), label="Custom FFT")
        plt.plot(np.abs(numpy_fft_result), "--", label="NumPy FFT")
        plt.title(f"FFT Comparison\nRel Error: {rel_error:.2e}, Corr: {corr:.4f}")
        plt.grid(True)
        plt.legend()

        print(f"Test Results for {name}:")
        print(f"Relative Error: {rel_error:.2e}")
        print(f"Maximum Difference: {max_diff:.2e}")
        print(f"Correlation: {corr:.4f}")

    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    # Run the tests and save the plot
    fig = test_fft()
    fig.savefig("fft_verification.png")
    plt.close()
