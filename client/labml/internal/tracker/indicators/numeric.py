from abc import ABC
from typing import Dict, Optional

import numpy as np

from . import Indicator
from ...util.values import to_numpy


class NumericIndicator(Indicator, ABC):
    def get_mean(self) -> float:
        raise NotImplementedError()

    def get_histogram(self):
        raise NotImplementedError()

    def get_all_values(self):
        values = self.get_histogram()
        assert values is not None
        return values

    @property
    def mean_key(self):
        return f'{self.name}.mean'


class _Collection(NumericIndicator, ABC):
    def __init__(self, name: str, is_print: bool, options: Optional[Dict] = None):
        super().__init__(name=name, is_print=is_print, options=options)
        self._values = []

    def _merge(self):
        if len(self._values) == 0:
            return []
        elif len(self._values) == 1:
            return to_numpy(self._values[0]).ravel()
        else:
            values = [to_numpy(v).ravel() for v in self._values]
            merged = np.concatenate(values, axis=0)
            self._values = [merged]

            return merged

    def collect_value(self, value):
        self._values.append(value)

    def clear(self):
        self._values = []

    def is_empty(self) -> bool:
        return len(self._values) == 0

    def get_mean(self) -> float:
        return float(np.mean(self._merge()))

    def get_histogram(self):
        return self._merge()


class Histogram(_Collection):
    def copy(self, key: str):
        return Histogram(key, is_print=self.is_print, options=self.options)


class Scalar(_Collection):
    def get_histogram(self):
        return None

    def get_all_values(self):
        return self._merge()

    def copy(self, key: str):
        return Scalar(key, is_print=self.is_print, options=self.options)

    @property
    def mean_key(self):
        return f'{self.name}'
