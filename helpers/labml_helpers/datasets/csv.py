import pandas as pd
import torch

from typing import Callable, List

from torch.utils.data import TensorDataset


class CsvDataset(TensorDataset):
    data: pd
    y_cols: List
    x_cols: List
    transform: Callable
    test_fraction: float = 0.0
    train: bool

    def __init__(self, file_path: str, y_cols: List, x_cols: List, train: bool = True,
                 transform: Callable = lambda: None, test_fraction: float = 0.0, nrows: int = None):
        self.__dict__.update(**vars())

        self.data = pd.read_csv(**{'filepath_or_buffer': file_path, 'nrows': nrows})

        data_length = len(self.data)

        self.test_size = int(data_length * self.test_fraction)
        self.train_size = data_length - self.test_size

        self.train_data = self.data.iloc[0:self.train_size]
        self.test_data = self.data.iloc[self.train_size:]

        if train:
            x, y = torch.tensor(self.train_data[self.x_cols].values), torch.tensor(self.train_data[self.y_cols].values)
        else:
            x, y = torch.tensor(self.test_data[self.x_cols].values), torch.tensor(self.test_data[self.y_cols].values)

        super(CsvDataset, self).__init__(x, y)
