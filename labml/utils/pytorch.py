import torch

from labml import tracker
from labml.configs import BaseConfigs


def store_l1_l2(name: str, tensor: torch.Tensor):
    tracker.add(f"{name}.mean", tensor.mean())
    tracker.add(f"{name}.l1", tensor.abs().mean())
    tracker.add(f"{name}.l2", (tensor ** 2).mean().sqrt())


def store_model_indicators(model: torch.nn.Module, model_name: str = "model"):
    for name, param in model.named_parameters():
        if param.requires_grad:
            with torch.no_grad():
                store_l1_l2(f"param.{model_name}.{name}", param)
                if param.grad is not None:
                    store_l1_l2(f"param.{model_name}.{name}.grad", param.grad)


def get_modules(configs: BaseConfigs):
    keys = dir(configs)

    modules = {}
    for k in keys:
        value = getattr(configs, k)
        if isinstance(value, torch.nn.Module):
            modules[k] = value

    return modules


def get_device(module: torch.nn.Module):
    params = module.parameters()
    try:
        sample_param = next(params)
        return sample_param.device
    except StopIteration:
        raise RuntimeError(f"Unable to determine"
                           f" device of {module.__class__.__name__}") from None
