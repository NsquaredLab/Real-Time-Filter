import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from rtfilter.butterworth import Butterworth, FilterType

# Set numpy random seed generator
seed = 5
random_generator = np.random.default_rng(seed)


def main():
    number_of_channels = 3
    # Initialize x values
    sampling_frequency = 2000
    time = 10  # seconds
    number_of_samples = time * sampling_frequency
    x = np.linspace(0, time, number_of_samples)

    # Initialize random frequencies
    frequencies = [
        random_generator.integers(
            low=random_generator.integers(10, 100),
            high=random_generator.integers(500, 1000),
            size=random_generator.integers(50, 100),
        )
        for _ in range(number_of_channels)
    ]

    # Add Frequencies 50, 100, 150, 200, 250 to it if not already included
    additional_frequencies = [50, 100, 150, 200, 250]
    for channel in range(number_of_channels):
        for freq in additional_frequencies:
            if freq not in frequencies[channel]:
                frequencies[channel] = np.append(frequencies[channel], freq)

    # Remove duplicates
    frequencies = [np.unique(freqs) for freqs in frequencies]

    # Sort
    frequencies = [np.sort(freqs) for freqs in frequencies]

    # Initialize y values
    y = np.zeros((number_of_channels, number_of_samples))
    for channel in range(number_of_channels):
        for f in frequencies[channel]:
            y[channel] += 1 * np.sin(2 * np.pi * f * x)

    # Initialize Butterworth filter
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
        "center_freq": additional_frequencies,
        "fs": sampling_frequency,
    }
    notch = Butterworth(number_of_channels, filter_type, filter_params)

    # Apply bandpass filter
    y_band_filtered = bandpass.filter(y)

    # Apply notch filter
    y_notch_filtered = notch.filter(y_band_filtered)

    # Plot
    figure = plt.figure()
    grid = GridSpec(1, number_of_channels, figure=figure)

    for channel in range(number_of_channels):
        ax = figure.add_subplot(grid[0, channel])
        ax.plot(x, y[channel], label="Original Signal")
        ax.plot(x, y_band_filtered[channel], label="Bandpass Filtered Signal")
        ax.plot(x, y_notch_filtered[channel], label="Notch Filtered Signal")
        ax.legend()

    plt.tight_layout()

    # Calculate FFT
    fft_y = np.fft.fft(y)
    fft_y_band_filtered = np.fft.fft(y_band_filtered)
    fft_y_notch_filtered = np.fft.fft(y_notch_filtered)

    # Calculate frequency values
    freqs = np.fft.fftfreq(number_of_samples, 1 / sampling_frequency)

    # Plot FFT
    fft_figure = plt.figure()
    grid = GridSpec(number_of_channels, 3, figure=fft_figure)

    for channel in range(number_of_channels):
        ax1 = fft_figure.add_subplot(grid[channel, 0])
        ax1.plot(
            freqs[: number_of_samples // 2],
            np.abs(fft_y[channel])[: number_of_samples // 2],
        )
        ax1.set_title("Original Signal")

        ax2 = fft_figure.add_subplot(grid[channel, 1])
        ax2.plot(
            freqs[: number_of_samples // 2],
            np.abs(fft_y_band_filtered[channel])[: number_of_samples // 2],
        )
        ax2.set_title("Bandpass Filtered Signal")

        ax3 = fft_figure.add_subplot(grid[channel, 2])
        ax3.plot(
            freqs[: number_of_samples // 2],
            np.abs(fft_y_notch_filtered[channel])[: number_of_samples // 2],
        )
        ax3.set_title("Notch Filtered Signal")


if __name__ == "__main__":
    main()
    plt.show()
