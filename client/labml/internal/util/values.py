import numpy as np


def _torch_to_numpy(tensor):
    import torch
    with torch.no_grad():
        tensor = tensor.cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float)

        return tensor.numpy()


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
            return _torch_to_numpy(value.data)
        elif isinstance(value, torch.Tensor):
            return _torch_to_numpy(value)

    try:
        import jaxlib
    except ImportError:
        jaxlib = None

    if jaxlib is not None:
        from jaxlib.xla_extension import DeviceArray
        if isinstance(value, DeviceArray):
            return np.asarray(value)

    raise ValueError(f"Unknown type {type(value)}")
