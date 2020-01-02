import torch

from .indicators import Histogram
from .. import logger


def add_model_indicators(model: torch.nn.Module, model_name: str = "model"):
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.add_indicator(Histogram(f"{model_name}.{name}"))
            logger.add_indicator(Histogram(f"{model_name}.{name}.grad"))


def store_model_indicators(model: torch.nn.Module, model_name: str = "model"):
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.store(f"{model_name}.{name}", param.data.cpu().numpy())
            logger.store(f"{model_name}.{name}.grad", param.grad.cpu().numpy())
