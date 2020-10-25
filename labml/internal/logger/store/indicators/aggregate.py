import typing

from labml.utils.pytorch import store_model_indicators

from . import Indicator

if typing.TYPE_CHECKING:
    import torch


class PyTorchModule(Indicator):
    def __init__(self, name: str, is_print=False):
        super().__init__(name=name, is_print=is_print)

    def is_empty(self) -> bool:
        return True

    def collect_value(self, model: 'torch.nn.Module'):
        store_model_indicators(model, self.name)

    def get_histogram(self):
        return None

    def copy(self, key: str):
        return PyTorchModule(key, is_print=self.is_print)
