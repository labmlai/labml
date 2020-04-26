import torch

from lab import logger, tracker
from lab.configs import BaseConfigs
from lab.logger import Text


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


def get_device(use_cuda: bool, cuda_device: int):
    is_cuda = use_cuda and torch.cuda.is_available()
    if not is_cuda:
        return torch.device('cpu')
    else:
        if cuda_device < torch.cuda.device_count():
            return torch.device('cuda', cuda_device)
        else:
            logger.log(f"Cuda device index {cuda_device} higher than "
                       f"device count {torch.cuda.device_count()}", Text.warning)
            return torch.device('cuda', torch.cuda.device_count() - 1)


def get_modules(configs: BaseConfigs):
    keys = dir(configs)

    modules = {}
    for k in keys:
        value = getattr(configs, k)
        if isinstance(value, torch.nn.Module):
            modules[k] = value

    return modules
