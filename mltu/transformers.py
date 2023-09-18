import logging
import typing
import numpy as np


class Transformer:
    def __init__(self, log_level: int = logging.INFO) -> None:
        self._log_level = log_level

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def __call__(self, data: typing.Any, label: typing.Any, *args, **kwargs):
        raise NotImplementedError


class LabelIndexer(Transformer):
    """Convert label to index by vocab

    Attributes:
        vocab (typing.List[str]): List of characters in vocab
    """

    def __init__(
            self,
            vocab: typing.List[str]
    ) -> None:
        self.vocab = vocab

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return data, np.array([self.vocab.index(l) for l in label if l in self.vocab])


class LabelPadding(Transformer):
    """Pad label to max_word_length

    Attributes:
        max_word_length (int): Maximum length of label
        padding_value (int): Value to pad
    """

    def __init__(
            self,
            max_word_length: int,
            padding_value: int
    ) -> None:
        self.max_word_length = max_word_length
        self.padding_value = padding_value

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return data, np.pad(label, (0, self.max_word_length - len(label)), "constant",
                            constant_values=self.padding_value)


class SpectrogramPadding(Transformer):
    """Pad spectrogram to max_spectrogram_length

    Attributes:
        max_spectrogram_length (int): Maximum length of spectrogram
        padding_value (int): Value to pad
    """

    def __init__(
            self,
            max_spectrogram_length: int,
            padding_value: int
    ) -> None:
        self.max_spectrogram_length = max_spectrogram_length
        self.padding_value = padding_value

    def __call__(self, spectrogram: np.ndarray, label: np.ndarray):
        padded_spectrogram = np.pad(spectrogram, ((0, self.max_spectrogram_length - spectrogram.shape[0]), (0, 0)),
                                    mode="constant", constant_values=self.padding_value)

        return padded_spectrogram, label
