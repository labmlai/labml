import numpy as np

try:
    import torch
except ImportError:
    torch = None


def to_numpy(value):
    if isinstance(value, int) or isinstance(value, float):
        return np.array(value)
    elif isinstance(value, np.number):
        return np.array(value.item())
    elif isinstance(value, list):
        return np.array(value)
    elif isinstance(value, np.ndarray):
        return value
    elif torch is not None:
        if isinstance(value, torch.nn.parameter.Parameter):
            return value.data.cpu().numpy()
        elif isinstance(value, torch.Tensor):
            return value.data.cpu().numpy()

    raise ValueError(f"Unknown type {type(value)}")
