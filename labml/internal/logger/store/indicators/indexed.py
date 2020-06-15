from abc import ABC

import numpy as np
from .numeric import NumericIndicator

try:
    import torch
except ImportError:
    torch = None


class IndexedIndicator(NumericIndicator, ABC):
    def __init__(self, name: str):
        super().__init__(name=name, is_print=False)
        self._values = []
        self._indexes = []

    def clear(self):
        self._values = []
        self._indexes = []

    def collect_value(self, value):
        if type(value) == tuple:
            assert len(value) == 2
            if type(value[0]) == int:
                self._indexes.append(value[0])
                self._values.append(value[1])
            else:
                assert type(value[0]) == list
                assert len(value[0]) == len(value[1])
                self._indexes += value[0]
                self._values += value[1]
        else:
            assert type(value) == list
            self._indexes += [v[0] for v in value]
            self._values += [v[1] for v in value]

    def is_empty(self) -> bool:
        return len(self._values) == 0

    def get_mean(self) -> float:
        return float(np.mean(self._values))

    def get_index_mean(self):
        summary = {}
        for ind, values in zip(self._indexes, self._values):
            if ind not in summary:
                summary[ind] = []
            summary[ind].append(values)

        indexes = []
        means = []
        for ind, values in summary.items():
            indexes.append(ind)
            means.append(float(np.mean(values)))

        return indexes, means


class IndexedScalar(IndexedIndicator):
    def get_histogram(self):
        return None

    def copy(self, key: str):
        return IndexedScalar(key)
