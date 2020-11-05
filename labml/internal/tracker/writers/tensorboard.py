from pathlib import PurePath
from typing import Dict

from ..indicators import Indicator
from ..indicators.artifacts import Image
from ..indicators.numeric import NumericIndicator

from . import Writer as WriteBase
from labml.internal.util.tensorboard_writer import get_tensorboard_writer


class Writer(WriteBase):
    def __init__(self, log_path: PurePath):
        super().__init__()

        self.__writer = get_tensorboard_writer(log_path)

    @staticmethod
    def _parse_key(key: str):
        return key.replace('.', '/')

    def write_h_parameters(self, hparams: Dict[str, any]):
        self.__writer.hparams(hparams)

    def _write_indicator(self, global_step: int, indicator: Indicator):
        if indicator.is_empty():
            return

        if isinstance(indicator, NumericIndicator):
            hist_data = indicator.get_histogram()
            if hist_data is not None:
                self.__writer.add_histogram(self._parse_key(indicator.name), hist_data, global_step)

            self.__writer.add_scalar(self._parse_key(indicator.mean_key), indicator.get_mean(), global_step)

        if isinstance(indicator, Image):
            for key in indicator.keys():
                self.__writer.add_image(self._parse_key(indicator.name), indicator.get_value(key), global_step)

    def write(self, *,
              global_step: int,
              indicators: Dict[str, Indicator]):
        for ind in indicators.values():
            self._write_indicator(global_step, ind)
