from pathlib import PurePath
from typing import Dict

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from . import TensorboardWriter

tf.config.experimental.set_visible_devices([], "GPU")


class TensorflowTensorboardWriter(TensorboardWriter):
    def __init__(self, log_path: PurePath):
        super().__init__()

        self.__log_path = log_path
        self.__writer = None

    def __connect(self):
        if self.__writer is not None:
            return

        self.__writer = tf.summary.create_file_writer(str(self.__log_path))

    def hparams(self, hparams: Dict[str, any]):
        self.__connect()
        with self.__writer.as_default():
            hp.hparams(hparams)

    def add_scalar(self, key: str, value: any, global_step: int):
        self.__connect()
        with self.__writer.as_default():
            tf.summary.scalar(key, value, step=global_step)

    def add_histogram(self, key: str, value: any, global_step: int):
        self.__connect()
        with self.__writer.as_default():
            tf.summary.histogram(key, value, step=global_step)

    def add_image(self, key: str, img: any, global_step: int):
        self.__connect()
        img = img.cpu().detach().permute(0, 2, 3, 1).numpy()
        with self.__writer.as_default():
            tf.summary.image(key, img, step=global_step)
