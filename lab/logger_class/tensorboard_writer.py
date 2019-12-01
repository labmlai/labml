from pathlib import PurePath

import numpy as np

import lab.logger_class.writers
import tensorflow as tf


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

    def write(self, *, global_step: int,
              queues,
              histograms,
              pairs,
              scalars):
        self.__connect()

        with self.__writer.as_default():
            for k, v in queues.items():
                if len(v) == 0:
                    continue
                tf.summary.histogram(self._parse_key(k), v, step=global_step)
                tf.summary.scalar(self._parse_key(f"{k}.mean"), float(np.mean(v)), step=global_step)

            for k, v in histograms.items():
                if len(v) == 0:
                    continue
                tf.summary.histogram(self._parse_key(k), v, step=global_step)
                tf.summary.scalar(self._parse_key(f"{k}.mean"), float(np.mean(v)), step=global_step)

            for k, v in scalars.items():
                if len(v) == 0:
                    continue
                tf.summary.scalar(self._parse_key(k), v, step=global_step)
