# import os
from pathlib import PurePath
from typing import Dict

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from . import Writer as WriteBase
from lab.internal.logger.store.artifacts import Artifact, Image
from lab.internal.logger.store.indicators import Indicator

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

            for art in artifacts.values():
                if art.is_empty():
                    continue

                if type(art) == Image:
                    # Expecting NxCxHxW images in pytorch tensors normalized in [0, 1]
                    for key in art.keys():
                        img = art.get_value(key)
                        img = img.cpu().detach().permute(0, 2, 3, 1).numpy()
                        tf.summary.image(self._parse_key(art.name),
                                         img,
                                         step=global_step)
