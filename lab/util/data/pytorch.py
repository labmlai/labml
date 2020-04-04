import pandas as pd

from typing import Callable, List

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    data: pd
    y_cols: List
    x_cols: List
    transform: Callable
    test_fraction: float = 0.0
    train: bool


class CsvDataset(BaseDataset):
    def __init__(self, file_path: str, y_cols: List, x_cols: List, train: bool = True,
                 transform: Callable = lambda: None, test_fraction: float = 0.0):
        self.__dict__.update(**vars())

        self.data = pd.read_csv(file_path)

        data_length = len(self.data)

        self.test_size = int(data_length * self.test_fraction)
        self.train_size = data_length - self.test_size

        self.train_data = self.data.iloc[0:self.train_size]
        self.test_data = self.data.iloc[self.train_size:]

    def __len__(self):
        if self.train:
            return self.train_size
        else:
            return self.test_size

    def __getitem__(self, index):
        if self.train:
            x, y = self.train_data.iloc[index][self.x_cols].values, self.train_data.iloc[index][self.y_cols].values
        else:
            x, y = self.test_data.iloc[index][self.y_cols].values, self.test_data.iloc[index][self.y_cols].values

        return x, y
