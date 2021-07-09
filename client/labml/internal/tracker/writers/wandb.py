from pathlib import Path
from typing import Dict, Any

import wandb

from labml.internal.configs.processor import ConfigsSaver
from . import Writer as WriteBase
from ..indicators import Indicator
from ..indicators.numeric import NumericIndicator


class WandBConfigsSaver(ConfigsSaver):
    def __init__(self, wandb_: 'wandb'):
        self.wandb = wandb_

    def save(self, configs: Dict[str, Any]):
        values = {}
        for k, v in configs.items():
            if v['order'] < 0:
                continue
            if v['value'] is not None:
                values[k] = v['value']
            elif v['computed'] is not None:
                values[k] = v['computed']
        self.wandb.config.update(values, allow_val_change=True)


class Writer(WriteBase):
    def __init__(self):
        super().__init__()
        self.wandb = wandb
        self.configs_saver = None
        self.run = None

    def init(self, name: str, log_dir: Path):
        self.run = self.wandb.init(project=name, dir=str(log_dir))

    @staticmethod
    def _parse_key(key: str):
        return key

    # def write_h_parameters(self, hparams: Dict[str, any]):
    #     self.wandb.config.update(hparams)

    def get_configs_saver(self):
        if self.configs_saver is None:
            self.configs_saver = WandBConfigsSaver(self.wandb)
        return self.configs_saver

    def _write_indicator(self, global_step: int, indicator: Indicator):
        if indicator.is_empty():
            return

        if isinstance(indicator, NumericIndicator):
            hist_data = indicator.get_histogram()
            if hist_data is not None:
                self.wandb.log({self._parse_key(indicator.name): self.wandb.Histogram(hist_data)},
                               step=global_step)

            self.wandb.log({self._parse_key(indicator.mean_key): indicator.get_mean()},
                           step=global_step)

    def write(self, *,
              global_step: int,
              indicators: Dict[str, Indicator]):
        for ind in indicators.values():
            self._write_indicator(global_step, ind)
