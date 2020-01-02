from typing import Dict

import numpy as np

from ..colors import Text
from ..indicators import Indicator


class Writer:
    def write(self, *,
              global_step: int,
              values: Dict[str, any],
              indicators: Dict[str, Indicator]):
        raise NotImplementedError()


class ScreenWriter(Writer):
    def __init__(self, is_color=True):
        super().__init__()

        self.is_color = is_color
        self._estimates = {}
        self._beta = 0.9
        self._beta_pow = {}

    def update_estimate(self, k, v):
        if k not in self._estimates:
            self._estimates[k] = 0
            self._beta_pow[k] = 1.

        self._estimates[k] *= self._beta
        self._estimates[k] += (1 - self._beta) * v
        self._beta_pow[k] *= self._beta

    def get_empty_string(self, length, decimals):
        return ' ' * (length - 2 - decimals) + '-.' + '-' * decimals

    def get_value_string(self, k, v):
        if k not in self._estimates:
            assert v is None
            return self.get_empty_string(8, 2)

        estimate = self._estimates[k] / (1 - self._beta_pow[k])
        lg = int(np.floor(np.log10(estimate))) + 1

        decimals = 7 - lg
        decimals = max(1, decimals)
        decimals = min(6, decimals)

        fmt = "{v:8." + str(decimals) + "f}"
        if v is None:
            return self.get_empty_string(8, decimals)
        else:
            return fmt.format(v=v)

    def write(self, *,
              global_step: int,
              values: Dict[str, any],
              indicators: Dict[str, Indicator]):
        parts = []

        for k, ind in indicators.items():
            if not ind.options.is_print:
                continue

            parts.append((f" {k}: ", None))

            if len(values[k]) == 0:
                value = self.get_value_string(k, None)
            else:
                v = np.mean(values[k])
                self.update_estimate(k, v)
                value = self.get_value_string(k, v)

            if self.is_color:
                parts.append((value, Text.value))
            else:
                parts.append(value)

        return parts
