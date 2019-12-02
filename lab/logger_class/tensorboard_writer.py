from pathlib import PurePath
from typing import Dict

import numpy as np

import lab.logger_class.writers
import tensorflow as tf

from .indicators import Indicator, IndicatorType


class Writer(lab.logger_class.writers.Writer):
    def __init__(self, log_path: PurePath):
        super().__init__()

        self.__log_path = log_path
        self.__writer = None

    def __connect(self):
        if self.__writer is not None:
            return

        self.__writer = tf.summary.create_file_writer(str(self.__log_path))

    def _parse_key(self, key: str):
        return key.replace('.', '/')

    def write(self, *,
              global_step: int,
              values: Dict[str, any],
              indicators: Dict[str, Indicator]):
        self.__connect()

        with self.__writer.as_default():
            for k, ind in indicators.items():
                v = values[k]
                if len(v) == 0:
                    continue
                if ind.type_ == IndicatorType.queue or ind.type_ == IndicatorType.histogram:
                    tf.summary.histogram(self._parse_key(k), v, step=global_step)

                if ind.type_ != IndicatorType.scalar:
                    key = self._parse_key(f"{k}.mean")
                else:
                    key = self._parse_key(f"{k}")

                tf.summary.scalar(key, float(np.mean(v)), step=global_step)

