from pathlib import PurePath
from typing import Dict


class TensorboardWriter:
    def hparams(self, hparams: Dict[str, any]):
        raise NotImplemented

    def add_scalar(self, key: str, value: any, global_step: int):
        raise NotImplemented

    def add_histogram(self, key: str, value: any, global_step: int):
        raise NotImplemented

    def add_image(self, key: str, img: any, global_step: int):
        raise NotImplemented


def get_tensorboard_writer(path: PurePath) -> TensorboardWriter:
    try:
        import tensorboard
    except ImportError as e:
        raise e

    try:
        from .pytorch import PyTorchTensorboardWriter

        return PyTorchTensorboardWriter(path)
    except ImportError:
        pass

    try:
        from .tensorflow import TensorflowTensorboardWriter

        return TensorflowTensorboardWriter(path)
    except ImportError:
        raise ImportError("Tensorboard writer requires pytorch or tensorflow installed.\n"
                          "Install pytorch with pip install torch. Visit pytorch.org for details.\n"
                          "Install tensorflow with pip install tensorflow. Visit tensorflow.org for details")
