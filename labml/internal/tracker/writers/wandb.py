from typing import Dict

import numpy as np
import wandb

from . import Writer as WriteBase
from ..indicators import Indicator
from ..indicators.numeric import NumericIndicator


class Writer(WriteBase):
    def __init__(self):
        super().__init__()
        self.wandb = wandb
        self.wandb.init()

    @staticmethod
    def _parse_key(key: str):
        return key

    def write_h_parameters(self, hparams: Dict[str, any]):
        self.wandb.config.update(hparams)

    def _write_indicator(self, global_step: int, indicator: Indicator):
        if indicator.is_empty():
            return

        if isinstance(indicator, NumericIndicator):
            hist_data = indicator.get_histogram()
            if hist_data is not None:
                self.wandb.log({self._parse_key(indicator.name): self.wandb.Histogram(hist_data),
                                'custom_step': global_step,
                                })

            self.wandb.log({self._parse_key(indicator.mean_key): indicator.get_mean(),
                            'custom_step': global_step,
                            })

    def write(self, *,
              global_step: int,
              indicators: Dict[str, Indicator]):
        for ind in indicators.values():
            self._write_indicator(global_step, ind)
