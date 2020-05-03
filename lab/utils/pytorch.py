import torch

from lab import tracker
from lab.configs import BaseConfigs


def add_model_indicators(model: torch.nn.Module, model_name: str = "model"):
    for name, param in model.named_parameters():
        if param.requires_grad:
            tracker.set_histogram(f"{model_name}.{name}")
            tracker.set_histogram(f"{model_name}.{name}.grad")


def store_model_indicators(model: torch.nn.Module, model_name: str = "model"):
    for name, param in model.named_parameters():
        if param.requires_grad:
            tracker.add(f"{model_name}.{name}", param)
            tracker.add(f"{model_name}.{name}.grad", param.grad)


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
