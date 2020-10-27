from collections import deque
from pathlib import PurePath
from typing import Dict, Optional

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from . import TensorboardWriter


class PyTorchTensorboardWriter(TensorboardWriter):
    __writer: Optional[SummaryWriter]

    def __init__(self, log_path: PurePath):
        super().__init__()

        self.__log_path = log_path
        self.__writer = None

    def __connect(self):
        if self.__writer is not None:
            return

        self.__writer = SummaryWriter(str(self.__log_path))

    def hparams(self, hparams: Dict[str, any]):
        self.__connect()

    def add_scalar(self, key: str, value: any, global_step: int):
        self.__connect()
        self.__writer.add_scalar(key, value, global_step)

    def add_histogram(self, key: str, value: any, global_step: int):
        self.__connect()
        if isinstance(value, list) or isinstance(value, deque):
            value = np.array(value)
        self.__writer.add_histogram(key, value, global_step)

    def add_image(self, key: str, img: any, global_step: int):
        self.__connect()
        self.__writer.add_image(key, img, global_step)
