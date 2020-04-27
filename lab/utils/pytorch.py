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
