from typing import Dict, Any, Optional

import comet_ml
from labml.internal.configs.processor import ConfigsSaver

from . import Writer as WriteBase
from ..indicators import Indicator
from ..indicators.numeric import NumericIndicator


class CometConfigsSaver(ConfigsSaver):
    def __init__(self, run: 'comet_ml.Experiment'):
        self.run = run

    def save(self, configs: Dict[str, Any]):
        values = {}
        for k, v in configs.items():
            if v['order'] < 0:
                continue
            if v['value'] is not None:
                values[k] = v['value']
            elif v['computed'] is not None:
                values[k] = v['computed']
        self.run.log_parameters(values)


class Writer(WriteBase):
    def __init__(self):
        super().__init__()
        self.comet = comet_ml
        self.configs_saver = None
        self.run: Optional['comet_ml.Experiment'] = None

    def init(self, name: str):
        self.run = self.comet.Experiment(project_name=name)

    @staticmethod
    def _parse_key(key: str):
        return key

    # def write_h_parameters(self, hparams: Dict[str, any]):
    #     self.wandb.config.update(hparams)

    def get_configs_saver(self):
        if self.configs_saver is None:
            self.configs_saver = CometConfigsSaver(self.run)
        return self.configs_saver

    def _write_indicator(self, global_step: int, indicator: Indicator):
        if indicator.is_empty():
            return

        if isinstance(indicator, NumericIndicator):
            self.run.log_metrics({self._parse_key(indicator.mean_key): indicator.get_mean()},
                                 step=global_step)

    def write(self, *,
              global_step: int,
              indicators: Dict[str, Indicator]):
        for ind in indicators.values():
            self._write_indicator(global_step, ind)
