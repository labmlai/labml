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

    def write(self, *,
              global_step: int,
              values: Dict[str, any],
              indicators: Dict[str, Indicator]):
        parts = []

        for k, ind in indicators.items():
            if not ind.options.is_print:
                continue

            if len(values[k]) == 0:
                continue

            v = np.mean(values[k])

            parts.append((f" {k}: ", None))
            if self.is_color:
                parts.append((f"{v :8,.2f}", Text.value))
            else:
                parts.append((f"{v :8,.2f}", None))

        return parts
