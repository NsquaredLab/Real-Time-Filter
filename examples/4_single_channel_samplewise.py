import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from rtfilter.butterworth import Butterworth, FilterType

# Set numpy random seed generator
seed = 5
random_generator = np.random.default_rng(seed)


def main():
    # Initialize x values
    sampling_frequency = 2000
    time = 10  # seconds
    number_of_samples = time * sampling_frequency
    x = np.linspace(0, time, number_of_samples)

    # Initialize random frequencies
    frequencies = random_generator.integers(low=1, high=1000, size=100)

    # Add Frequencies 50, 100, 150, 200, 250 to it if not already included
    additional_frequencies = [50, 100, 150, 200, 250]
    for freq in additional_frequencies:
        if freq not in frequencies:
            frequencies = np.append(frequencies, freq)

    # Remove duplicates
    frequencies = np.unique(frequencies)
    amplitudes = random_generator.random(size=frequencies.shape[0]) * 20

    # Sort
    frequencies = np.sort(frequencies)

    # Initialize y values
    y = np.zeros_like(x)
    for freq in frequencies:
        y += 1 * np.sin(2 * np.pi * freq * x)

    # Initialize Butterworth filter
    number_of_channels = 1
    filter_type = FilterType.Bandpass
    filter_params = {
        "order": 4,
        "lowcut": 100,
        "highcut": 500,
        "fs": sampling_frequency,
    }

    bandpass = Butterworth(number_of_channels, filter_type, filter_params)

    filter_type = FilterType.Notch
    filter_params = {
        "center_freq": 50,
        "fs": sampling_frequency,
    }
    notch = Butterworth(number_of_channels, filter_type, filter_params)

    y_band_filtered = np.zeros_like(y)
    y_notch_filtered = np.zeros_like(y)
    # Apply bandpass filter
    for sample in range(number_of_samples):
        y_band_filtered[sample] = bandpass.filter(y[sample], multiple_samples=False)
        # Apply notch filter
        y_notch_filtered[sample] = notch.filter(
            y_band_filtered[sample], multiple_samples=False
        )

    # Plot
    plt.figure()
    plt.plot(x, y, label="Original Signal")
    plt.plot(x, y_band_filtered, label="Bandpass Filtered Signal")
    plt.plot(x, y_notch_filtered, label="Notch Filtered Signal")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    # Calculate FFT
    fft_y = np.fft.fft(y)
    fft_y_band_filtered = np.fft.fft(y_band_filtered)
    fft_y_notch_filtered = np.fft.fft(y_notch_filtered)

    # Calculate frequency values
    freqs = np.fft.fftfreq(number_of_samples, 1 / sampling_frequency)

    # Plot FFT
    fft_figure = plt.figure()
    grid = GridSpec(1, 3, figure=fft_figure)

    ax1 = fft_figure.add_subplot(grid[0, 0])
    ax1.plot(freqs[: number_of_samples // 2], np.abs(fft_y)[: number_of_samples // 2])
    ax1.set_title("Original Signal")

    ax2 = fft_figure.add_subplot(grid[0, 1])
    ax2.plot(
        freqs[: number_of_samples // 2],
        np.abs(fft_y_band_filtered)[: number_of_samples // 2],
    )
    ax2.set_title("Bandpass Filtered Signal")

    ax3 = fft_figure.add_subplot(grid[0, 2])
    ax3.plot(
        freqs[: number_of_samples // 2],
        np.abs(fft_y_notch_filtered)[: number_of_samples // 2],
    )
    ax3.set_title("Notch Filtered Signal")

    plt.tight_layout()


if __name__ == "__main__":
    main()
    plt.show()
