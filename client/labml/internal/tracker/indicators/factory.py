from typing import Dict

import numpy as np

from .numeric import Histogram, Scalar


def load_indicator_from_dict(data: Dict[str, any]):
    class_name = data['class_name']
    del data['class_name']

    if class_name == 'Histogram':
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

    try:
        import torch

        if isinstance(value, torch.nn.parameter.Parameter):
            return Scalar(name, is_print)
        elif isinstance(value, torch.Tensor):
            return Scalar(name, is_print)
    except ImportError:
        pass

    try:
        import jaxlib
        from jaxlib.xla_extension import DeviceArray
        if isinstance(value, DeviceArray):
            return Scalar(name, is_print)
    except ImportError:
        pass

    raise ValueError(f"Unknown type {type(value)}")
