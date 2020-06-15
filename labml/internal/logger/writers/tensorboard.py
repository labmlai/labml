from pathlib import PurePath
from typing import Dict

import tensorflow as tf
from labml.internal.logger.store.indicators import Indicator
from labml.internal.logger.store.indicators.artifacts import Image
from labml.internal.logger.store.indicators.numeric import NumericIndicator
from tensorboard.plugins.hparams import api as hp

from . import Writer as WriteBase

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

    def write_h_parameters(self, hparams: Dict[str, any]):
        self.__connect()

        with self.__writer.as_default():
            hp.hparams(hparams)

    def _write_indicator(self, global_step: int, indicator: Indicator):
        if indicator.is_empty():
            return

        if isinstance(indicator, NumericIndicator):
            hist_data = indicator.get_histogram()
            if hist_data is not None:
                tf.summary.histogram(self._parse_key(indicator.name),
                                     hist_data, step=global_step)

            tf.summary.scalar(self._parse_key(indicator.mean_key),
                              indicator.get_mean(),
                              step=global_step)

        if isinstance(indicator, Image):
            for key in indicator.keys():
                img = indicator.get_value(key)
                img = img.cpu().detach().permute(0, 2, 3, 1).numpy()
                tf.summary.image(self._parse_key(indicator.name),
                                 img,
                                 step=global_step)

    def write(self, *,
              global_step: int,
              indicators: Dict[str, Indicator]):
        self.__connect()

        with self.__writer.as_default():
            for ind in indicators.values():
                self._write_indicator(global_step, ind)
