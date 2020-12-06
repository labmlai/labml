from typing import Dict

import numpy as np
from .numeric import Queue, Histogram, Scalar

try:
    import torch
except ImportError:
    torch = None


def load_indicator_from_dict(data: Dict[str, any]):
    class_name = data['class_name']
    del data['class_name']

    if class_name == 'Queue':
        return Queue(**data)
    elif class_name == 'Histogram':
        return Histogram(**data)
    elif class_name == 'Scalar':
        return Scalar(**data)
    else:
        raise ValueError(f"Unknown indicator: {class_name}")


def create_default_indicator(name: str, value: any, is_print: bool):
    if isinstance(value, int) or isinstance(value, float):
        return Scalar(name, is_print)
    elif isinstance(value, np.number):
        return Scalar(name, is_print)
    elif isinstance(value, list):
        return Scalar(name, is_print)
    elif isinstance(value, np.ndarray):
        return Scalar(name, is_print)
    elif torch is not None:
        from torch.optim.optimizer import Optimizer
        if isinstance(value, torch.nn.parameter.Parameter):
            return Scalar(name, is_print)
        elif isinstance(value, torch.Tensor):
            return Scalar(name, is_print)
        elif isinstance(value, torch.nn.Module):
            from labml.internal.tracker.indicators.aggregate import PyTorchModule
            return PyTorchModule(name, is_print)
        elif isinstance(value, Optimizer):
            from labml.internal.tracker.indicators.aggregate import PyTorchOptimizer
            return PyTorchOptimizer(name, is_print)
        elif isinstance(value, tuple) and isinstance(value[0], Optimizer):
            from labml.internal.tracker.indicators.aggregate import PyTorchOptimizer
            return PyTorchOptimizer(name, is_print)

    raise ValueError(f"Unknown type {type(value)}")
