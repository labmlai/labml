import numpy as np


def to_numpy(value):
    if isinstance(value, int) or isinstance(value, float):
        return np.array(value)
    elif isinstance(value, np.number):
        return np.array(value.item())
    elif isinstance(value, list):
        return np.array(value)
    elif isinstance(value, np.ndarray):
        return value

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        if isinstance(value, torch.nn.parameter.Parameter):
            return value.data.cpu().numpy()
        elif isinstance(value, torch.Tensor):
            return value.data.cpu().numpy()

    try:
        import jaxlib
    except ImportError:
        jaxlib = None

    if jaxlib is not None:
        from jaxlib.xla_extension import DeviceArray
        if isinstance(value, DeviceArray):
            return np.asarray(value)

    raise ValueError(f"Unknown type {type(value)}")
