from enum import Enum, auto


class FilterType(Enum):
    Bandpass = auto()
    Lowpass = auto()
    Highpass = auto()
    Notch = auto()
