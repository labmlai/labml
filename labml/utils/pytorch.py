from typing import Optional, Dict, TYPE_CHECKING

import torch

from labml import tracker
from labml.configs import BaseConfigs

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer


def store_l1_l2(name: str, tensor: torch.Tensor):
    if tensor.is_floating_point():
        tracker.add(f"{name}.mean", tensor.mean())
        tracker.add(f"{name}.l1", tensor.abs().mean())
        tracker.add(f"{name}.l2", (tensor ** 2).mean().sqrt())


def store_model_indicators(model: torch.nn.Module, *, model_name: str = "model"):
    for name, param in model.named_parameters():
        if param.requires_grad:
            with torch.no_grad():
                store_l1_l2(f"param.{model_name}.{name}", param)
                if param.grad is not None:
                    store_l1_l2(f"grad.{model_name}.{name}", param.grad)


def store_optimizer_indicators(optimizer: 'Optimizer', *,
                               models: Optional[Dict[str, torch.nn.Module]] = None,
                               optimizer_name: str = "optimizer"):
    if models is None:
        models = {}
    names = {}
    for model_name, model in models.items():
        for name, p in model.named_parameters():
            names[p] = f'{model_name}.{name}'

    unknown = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            state = optimizer.state[p]
            if len(state) == 0:
                continue

            name = names.get(p, None)
            if name is None:
                name = f'unknown.{unknown}'
                unknown += 1

            for k, v in state.items():
                if isinstance(v, float) or isinstance(v, int):
                    tracker.add(f'optim.{optimizer_name}.{name}.{k}', v)
                if isinstance(v, torch.Tensor):
                    store_l1_l2(f'optim.{optimizer_name}.{name}.{k}', v)


def get_modules(configs: BaseConfigs):
    keys = dir(configs)

    modules = {}
    for k in keys:
        type_ = configs._get_type(k)
        try:
            if issubclass(type_, torch.nn.Module):
                modules[k] = getattr(configs, k)
        except TypeError:
            pass

    return modules


def get_device(module: torch.nn.Module):
    params = module.parameters()
    try:
        sample_param = next(params)
        return sample_param.device
    except StopIteration:
        raise RuntimeError(f"Unable to determine"
                           f" device of {module.__class__.__name__}") from None
