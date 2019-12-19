import torch

from .indicators import IndicatorType, IndicatorOptions
from .. import logger


def add_model_indicators(model: torch.nn.Module, model_name: str = "model"):
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.add_indicator(f"{model_name}.{name}",
                                 IndicatorType.histogram,
                                 IndicatorOptions(is_print=False))
            logger.add_indicator(f"{model_name}.{name}.grad",
                                 IndicatorType.histogram,
                                 IndicatorOptions(is_print=False))


def store_model_indicators(model: torch.nn.Module, model_name: str = "model"):
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.store(f"{model_name}.{name}", param.data.cpu().numpy())
            logger.store(f"{model_name}.{name}.grad", param.grad.cpu().numpy())
