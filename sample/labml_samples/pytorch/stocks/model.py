from typing import List, Tuple

import torch
from labml_helpers.module import Module
from torch import nn

from labml_samples.pytorch.stocks import CandleIdx


class FrontPaddedConv1d(Module):
    def __init__(self, *, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size)

    def __call__(self, x: torch.Tensor):
        if self.kernel_size > 1:
            pad = x.new_zeros(*x.shape[:-1], self.kernel_size - 1)
            x = torch.cat([pad, x], dim=-1)
        return self.conv(x)


class CnnModel(Module):
    def __init__(self, *,
                 price_mean: float,
                 price_std: float,
                 volume_mean: float,
                 volume_std: float,
                 y_std: float,
                 y_mean: float,
                 activation: nn.Module,
                 conv_sizes: List[Tuple[int, int]],
                 dropout=0.1):
        super().__init__()

        self.y_mean = y_mean
        self.y_std = y_std
        self.volume_std = volume_std
        self.volume_mean = volume_mean
        self.price_std = price_std
        self.price_mean = price_mean
        layers = []
        in_channels = CandleIdx.all
        for s in conv_sizes:
            layers.append(FrontPaddedConv1d(in_channels=in_channels,
                                            out_channels=s[0],
                                            kernel_size=s[1]))
            in_channels = s[0]
        self.layers = nn.ModuleList(layers)

        self.final = FrontPaddedConv1d(in_channels=in_channels,
                                       out_channels=1,
                                       kernel_size=1)
        self.activation = activation

        self.conv_dropout = nn.Dropout(dropout)

    def __call__(self, x: torch.Tensor):
        x = x.permute((0, 2, 1))
        x[:, 0:CandleIdx.prices, :] = \
            (x[:, 0:CandleIdx.prices, :] - self.price_mean) / self.price_std
        x[:, CandleIdx.volume, :] = \
            (x[:, CandleIdx.prices, :] - self.volume_mean) / self.volume_std
        for layer in self.layers:
            x = self.activation(self.conv_dropout(layer(x)))
        x = self.final(x)

        x = x.permute((0, 2, 1))
        return x
