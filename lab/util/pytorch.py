import torch

from lab import logger
from lab.logger.colors import Text


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