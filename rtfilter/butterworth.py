from __future__ import annotations
from typing import TYPE_CHECKING, Union
import numpy as np
from rtfilter.core.filter_types import FilterType
from rtfilter.core.core_butterworth import Bandpass, Lowpass, Highpass, Notch


class Butterworth:
    def __init__(
        self,
        number_of_channels: int,
        filter_type: FilterType,
        filter_params: dict[str, Union[int, float]],
    ) -> None:
        """
        Initialize the real-time Butterworth filters.

        Args:
        """
        self._filters: Union[
            list[Union[Lowpass, Highpass, Bandpass, Notch]], list[list[Notch]]
        ] = []
        self._number_of_channels = number_of_channels
        self._filter_type = filter_type
        self._filter_params = filter_params

        self._notch_harmonics: bool = False

        self._initialize_filters()

    def filter(
        self,
        x: Union[float, np.ndarray[float]],
        multiple_samples: bool = True,
    ) -> Union[float, np.ndarray[float]]:
        """
        Filter a single input sample or a list of samples.
        """

        assert isinstance(
            x, (float, np.ndarray)
        ), f"The input x of type {type(x)} must be a float or np.ndarray."

        assert isinstance(
            multiple_samples, bool
        ), "The multiple_samples parameter must be a boolean."

        if isinstance(x, np.ndarray):
            assert x.ndim in [1, 2], "The input x must be a 1D or 2D np.ndarray."

            if x.ndim == 2:
                assert (
                    x.shape[0] == self._number_of_channels
                ), f"The input x must have the same number of channels as the filter. Expected {self._number_of_channels} channels, got {x.shape[0]} channels."

                if x.shape[1] == 1:
                    assert (
                        multiple_samples is False
                    ), f"x.shape[1] == 1 if multiple_samples are False. x.shape = {x.shape[1]} provided"

            x_filtered = np.zeros_like(x)
            if self._number_of_channels == 1:
                if self._filter_type == FilterType.Notch and self._notch_harmonics:
                    x_filtered = self._filter_notch_harmonics(x, 0, multiple_samples)
                else:
                    x_filtered = self._filters[0].apply_filter(x, multiple_samples)
            else:

                for channel in range(self._number_of_channels):
                    if self._filter_type == FilterType.Notch and self._notch_harmonics:
                        x_filtered[channel] = self._filter_notch_harmonics(
                            x[channel], channel, multiple_samples
                        )
                    else:
                        x_filtered[channel] = self._filters[channel].apply_filter(
                            x[channel], multiple_samples
                        )
            return np.array(x_filtered)

        if isinstance(x, float):
            assert (
                self._number_of_channels == 1
            ), f"If x of type float provided, number of channels should be 1. Number of channels provided: {self._number_of_channels}"

            assert (
                multiple_samples is False
            ), "Flag multiple samples should be False for input x of type float!"

            if self._filter_type == FilterType.Notch and self._notch_harmonics:
                x_filtered = self._filter_notch_harmonics(x, 0, multiple_samples)
            else:
                x_filtered = self._filters[0].apply_filter(x, multiple_samples)
            return float(x_filtered)

    def _filter_notch_harmonics(
        self,
        x: Union[float, np.ndarray[float]],
        channel: int,
        multiple_samples: bool = True,
    ) -> Union[float, np.ndarray[float]]:
        """
        Filter a single input sample or a list of samples with multiple notch filters.
        """
        for harmonic in self._filters[channel]:
            x = harmonic.apply_filter(x, multiple_samples)
        return x

    def _initialize_filters(self) -> None:
        """
        Initialize the Butterworth filters.
        """

        match self._filter_type:
            case FilterType.Bandpass:
                self._initialize_bandpass()
            case FilterType.Lowpass:
                self._initialize_lowpass()
            case FilterType.Highpass:
                self._initialize_highpass()
            case FilterType.Notch:
                self._initialize_notch()

    def _initialize_bandpass(self):
        """
        Initialize the bandpass filter.
        """
        param_keys = list(self._filter_params.keys())

        # Check optional parameters
        order = self._filter_params["order"] if "order" in param_keys else None

        assert "lowcut" in param_keys, "The lowcut parameter is required."
        assert "highcut" in param_keys, "The highcut parameter is required."
        assert "fs" in param_keys, "The fs parameter is required."

        for _ in range(self._number_of_channels):
            self._filters.append(
                Bandpass(
                    lowcut=self._filter_params["lowcut"],
                    highcut=self._filter_params["highcut"],
                    fs=self._filter_params["fs"],
                    order=order,
                )
            )

    def _initialize_lowpass(self):
        """
        Initialize the lowpass filter.
        """
        param_keys = list(self._filter_params.keys())

        # Check optional parameters
        order = self._filter_params["order"] if "order" in param_keys else None

        assert "cutoff" in param_keys, "The cutoff parameter is required."
        assert "fs" in param_keys, "The fs parameter is required."

        for _ in range(self._number_of_channels):
            self._filters.append(
                Lowpass(
                    cutoff=self._filter_params["cutoff"],
                    fs=self._filter_params["fs"],
                    order=order,
                )
            )

    def _initialize_highpass(self):
        """
        Initialize the highpass filter.
        """
        param_keys = list(self._filter_params.keys())

        # Check optional parameters
        order = self._filter_params["order"] if "order" in param_keys else None

        assert "cutoff" in param_keys, "The cutoff parameter is required."
        assert "fs" in param_keys, "The fs parameter is required."

        for _ in range(self._number_of_channels):
            self._filters.append(
                Highpass(
                    cutoff=self._filter_params["cutoff"],
                    fs=self._filter_params["fs"],
                    order=order,
                )
            )

    def _initialize_notch(self):
        """
        Initialize the notch filter.
        """
        param_keys = list(self._filter_params.keys())

        # Check optional parameters
        q = self._filter_params["q"] if "q" in param_keys else None

        assert (
            "center_freq" in param_keys
        ), "The center frequency parameter is required."
        assert "fs" in param_keys, "The fs parameter is required."

        center_freq = self._filter_params["center_freq"]

        if isinstance(center_freq, list):
            for channel in range(self._number_of_channels):
                self._filters.append([])
                for freq in center_freq:
                    (
                        self._filters[channel].append(
                            Notch(
                                center_freq=freq,
                                fs=self._filter_params["fs"],
                                q=self._filter_params["q"],
                            )
                        )
                        if "q" in param_keys
                        else self._filters[channel].append(
                            Notch(
                                center_freq=freq,
                                fs=self._filter_params["fs"],
                            )
                        )
                    )

            self._notch_harmonics = True

        if isinstance(center_freq, int) or isinstance(center_freq, float):
            for _ in range(self._number_of_channels):
                self._filters.append(
                    Notch(
                        center_freq=center_freq,
                        fs=self._filter_params["fs"],
                        q=q,
                    )
                    if "q" in param_keys
                    else self._filters.append(
                        Notch(
                            center_freq=center_freq,
                            fs=self._filter_params["fs"],
                        )
                    )
                )
