import numpy as np

try:
    import torch
except ImportError:
    torch = None


def to_numpy(value):
    type_ = type(value)

    if type_ == float or type_ == int:
        return np.array(value)

    if isinstance(value, np.number):
        return np.array(value.item())

    if type_ == list:
        return np.array(value)

    if type_ == np.ndarray:
        return value

    if torch is not None:
        if type_ == torch.nn.parameter.Parameter:
            return value.data.cpu().numpy()
        if type_ == torch.Tensor:
            return value.data.cpu().numpy()

    assert False, f"Unknown type {type_}"
