from typing import TYPE_CHECKING, Union, Tuple, Dict

from labml.utils.pytorch import store_model_indicators, store_optimizer_indicators

from . import Indicator

if TYPE_CHECKING:
    from torch.nn import Module
    from torch.optim.optimizer import Optimizer


class PyTorchModule(Indicator):
    def __init__(self, name: str, is_print=False):
        super().__init__(name=name, is_print=is_print)

    def is_empty(self) -> bool:
        return True

    def collect_value(self, model: 'Module'):
        store_model_indicators(model, model_name=self.name)

    def get_histogram(self):
        return None

    def copy(self, key: str):
        return PyTorchModule(key, is_print=self.is_print)


class PyTorchOptimizer(Indicator):
    def __init__(self, name: str, is_print=False):
        super().__init__(name=name, is_print=is_print)

    def is_empty(self) -> bool:
        return True

    def collect_value(self, optimizer: Union['Optimizer', Tuple['Optimizer', Dict[str, 'Module']]]):
        if isinstance(optimizer, tuple):
            store_optimizer_indicators(optimizer[0], models=optimizer[1], optimizer_name=self.name)
        else:
            store_optimizer_indicators(optimizer, optimizer_name=self.name)

    def get_histogram(self):
        return None

    def copy(self, key: str):
        return PyTorchModule(key, is_print=self.is_print)
