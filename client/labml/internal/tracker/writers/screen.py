from typing import Dict

import numpy as np

from labml.logger import Text
from .. import Writer, Indicator
from ..indicators import numeric


class ScreenWriter(Writer):
    def __init__(self):
        super().__init__()

        self._estimates = {}
        self._beta = 0.9
        self._beta_pow = {}
        self._last_printed_value = {}

    def update_estimate(self, k, v):
        if k not in self._estimates or not np.isfinite(self._estimates[k]):
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
        if abs(estimate) < 1e-9 or not np.isfinite(estimate):
            lg = 0
        else:
            lg = int(np.ceil(np.log10(abs(estimate)))) + 1

        if lg >= 7:
            fmt = "{v:9.3e}"
        else:
            decimals = np.clip(7 - lg, 1, 6)
            fmt = "{v:8,." + str(decimals) + "f}"

        if v is None:
            return self.get_empty_string(8, decimals)
        else:
            return fmt.format(v=v)

    @staticmethod
    def __format_artifact(length: int, value: str):
        fmt = "{v:>" + str(length + 1) + "}"
        return fmt.format(v=value)

    def _get_indicator_string(self, indicators: Dict[str, Indicator]):
        parts = []

        for ind in indicators.values():
            if not isinstance(ind, numeric.NumericIndicator):
                continue
            if not ind.is_print:
                continue

            parts.append((f" {ind.name}: ", None))

            if not ind.is_empty():
                v = ind.get_mean()
                self.update_estimate(ind.name, v)
                value = self.get_value_string(ind.name, v)
                self._last_printed_value[ind.name] = value
                parts.append((value, Text.value))
            elif ind.name in self._last_printed_value:
                value = self._last_printed_value[ind.name]
                parts.append((value, Text.subtle))
            else:
                value = self.get_value_string(ind.name, None)
                parts.append((value, Text.subtle))

        return parts

    def write(self, *,
              global_step: int,
              indicators: Dict[str, Indicator]):
        return self._get_indicator_string(indicators)
