# import os
from pathlib import PurePath
from typing import Dict

import tensorflow as tf

from . import Writer as WriteBase
from ..artifacts import Artifact
from ..indicators import Indicator

tf.config.experimental.set_visible_devices([], "GPU")


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Writer(WriteBase):
    def __init__(self, log_path: PurePath):
        super().__init__()

        self.__log_path = log_path
        self.__writer = None

    def __connect(self):
        if self.__writer is not None:
            return

        self.__writer = tf.summary.create_file_writer(str(self.__log_path))

    @staticmethod
    def _parse_key(key: str):
        return key.replace('.', '/')

    def write(self, *,
              global_step: int,
              indicators: Dict[str, Indicator],
              artifacts: Dict[str, Artifact]):
        self.__connect()

        with self.__writer.as_default():
            for ind in indicators.values():
                if ind.is_empty():
                    continue

                hist_data = ind.get_histogram()
                if hist_data is not None:
                    tf.summary.histogram(self._parse_key(ind.name), hist_data, step=global_step)

                mean_value = ind.get_mean()
                if mean_value is not None:
                    tf.summary.scalar(self._parse_key(ind.mean_key), mean_value,
                                      step=global_step)
