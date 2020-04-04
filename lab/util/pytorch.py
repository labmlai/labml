import torch
import pandas as pd

from typing import Callable, List

from lab import logger
from lab.configs import Configs
from lab.logger.colors import Text

from torch.utils.data import Dataset


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


def get_modules(configs: Configs):
    keys = dir(configs)

    modules = {}
    for k in keys:
        value = getattr(configs, k)
        if isinstance(value, torch.nn.Module):
            modules[k] = value

    return modules


class BaseDataset(Dataset):
    pass


class CustomDataset(BaseDataset):
    data: pd
    y_cols: List
    x_cols: List
    transform: Callable
    test_size: float = 0.0
    train: bool

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        data_length = len(self.data)

        self.test_size = int(data_length * self.test_size)
        self.train_size = data_length - self.test_size

        self.train_data = self.data.iloc[0:self.train_size]
        self.test_data = self.data.iloc[self.train_size:]

    @classmethod
    def from_csv(cls, file_path: str, train: bool, transform: Callable, test_size: float, y_cols: List, x_cols: List):
        data = pd.read_csv(file_path)

        kwargs = vars()
        kwargs['data'] = data

        return CustomDataset(**kwargs)

    def __len__(self):
        if self.train:
            return self.train_size
        else:
            return self.test_size

    def __getitem__(self, index):
        if self.train:
            x = self.train_data.iloc[index][self.x_cols].values
            y = self.train_data.iloc[index][self.y_cols].values
        else:
            x = self.test_data.iloc[index][self.y_cols].values
            y = self.test_data.iloc[index][self.y_cols].values

        return x, y
