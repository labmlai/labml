import torch

from labml import tracker
from labml.configs import BaseConfigs


def _store_l1_l2(name: str, tensor: torch.Tensor):
    tracker.add(f"{name}.mean", tensor.mean())
    tracker.add(f"{name}.l1", tensor.abs().mean())
    tracker.add(f"{name}.l2", (tensor ** 2).mean().sqrt())


def store_model_indicators(model: torch.nn.Module, model_name: str = "model"):
    for name, param in model.named_parameters():
        if param.requires_grad:
            with torch.no_grad():
                _store_l1_l2(f"param.{model_name}.{name}", param)
                if param.grad is not None:
                    _store_l1_l2(f"param.{model_name}.{name}.grad", param.grad)


class LogActivations:
    def __init__(self):
        self.entered = False

    def __enter__(self):
        self.entered = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.entered = False


_log_activations = LogActivations()


def log_activation():
    return _log_activations


class ForwardHook:
    def __init__(self, model_name, name: str, module: torch.nn.Module):
        self.model_name = model_name
        self.name = name
        self.module = module
        module.register_forward_hook(self)

    def __call__(self, module, i, o):
        if not _log_activations.entered:
            return

        if isinstance(o, torch.Tensor):
            _store_l1_l2(f"module.{self.model_name}.{self.name}", o)
        if isinstance(o, tuple):
            for i, t in enumerate(o):
                _store_l1_l2(f"module.{self.model_name}.{self.name}.{i}", t)


def hook_model_outputs(model: torch.nn.Module, model_name: str = "model"):
    for name, module in model.named_modules():
        if name == '':
            name = 'full'
        ForwardHook(model_name, name, module)


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
