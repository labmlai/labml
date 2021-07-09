import numpy as np
import torch
from labml import monit, lab
from labml import tracker
from torch.utils.data import Dataset

from labml_samples.pytorch.stocks import CandleIdx
from labml_samples.pytorch.stocks.build_numpy_cache import build_cache


def normalize_by_previous_day(packs: torch.Tensor):
    mean_price = packs[:, :, 0:CandleIdx.open_close].mean(dim=-1).mean(dim=-1)
    mean_volume = packs[:, :, CandleIdx.volume].mean(dim=-1)
    mean_price = mean_price.clamp_min(1e-6)
    mean_volume = mean_volume.clamp_min(1e-6)

    data = packs[1:, :, :] / mean_price[:-1].view(-1, 1, 1)
    data[:, :, CandleIdx.volume] = packs[1:, :, CandleIdx.volume] / mean_volume[:-1].view(-1, 1)

    return data.transpose_(0, 1)


def calculate_moving_average(data: torch.Tensor, start: int, end: int):
    cat = []
    pad = data.new_zeros(1, *data.shape[1:])
    cat.append(pad)
    if start <= 0:
        # pad = data.new_zeros(-start, *data.shape[1:])
        pad = data[:1].repeat(-start, *[1 for _ in data.shape[1:]])
        cat.append(pad)

    if start > 0:
        data = data[start:]
    if end < 0:
        data = data[:end]
    cat.append(data)

    if end > 0:
        # pad = data.new_zeros(end, *data.shape[1:])
        pad = data[-1:].repeat(end, *[1 for _ in data.shape[1:]])
        cat.append(pad)

    data = torch.cat(cat, dim=0)
    cum = data.cumsum(dim=0)

    range_ = end - start + 1

    avg = cum[range_:] - cum[:-range_]
    avg /= range_

    return avg


class MinutelyDataset(Dataset):
    def __init__(self, dates: np.ndarray, packets: torch.Tensor):
        self.data = normalize_by_previous_day(packets)
        self.data.log_()
        mean = self.data[:, :, 0:CandleIdx.open_close].mean(dim=-1)
        self.reference = calculate_moving_average(mean, -4, 0)
        self.target = calculate_moving_average(mean, 1, 5)
        self.strike_low = self.data[:, :, CandleIdx.low]
        self.strike_high = self.data[:, :, CandleIdx.high]
        self.strike_low[:-1] = self.data[1:, :, CandleIdx.low]
        self.strike_high[:-1] = self.data[1:, :, CandleIdx.high]

        self.data.transpose_(0, 1)
        self.reference.transpose_(0, 1).unsqueeze_(-1)
        self.target.transpose_(0, 1).unsqueeze_(-1)
        self.strike_low.transpose_(0, 1).unsqueeze_(-1)
        self.strike_high.transpose_(0, 1).unsqueeze_(-1)
        self.y = self.target - self.reference

        self.dates = dates[1:]

        self.price_std = self.data[:, :, 0:CandleIdx.prices].std().item()
        self.price_mean = self.data[:, :, 0:CandleIdx.prices].mean().item()
        self.volume_std = self.data[:, :, CandleIdx.volume].std().item()
        self.volume_mean = self.data[:, :, CandleIdx.volume].mean().item()
        self.y_std = self.y.std().item()
        self.y_mean = self.y.mean().item()

        tracker.set_tensor("ref.*", is_once=True)
        tracker.set_tensor("target.*", is_once=True)
        tracker.set_tensor("strike_low.*", is_once=True)
        tracker.set_tensor("strike_high.*", is_once=True)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {'data': self.data[idx],
                'target': self.y[idx]}

    def save_artifacts(self):
        tracker.save({'target.': self.y,
                      'ref.': self.reference,
                      'strike_low.': self.strike_low,
                      'strike_high.': self.strike_high})


class MinutelyData:
    def __init__(self, validation_dates: int, skip_cache: bool = False):
        self.validation_dates = validation_dates

        dates_cache_path = lab.get_data_path() / 'dates.npy'
        packets_cache_path = lab.get_data_path() / 'packets.npy'

        if skip_cache or not dates_cache_path.exists() or not packets_cache_path.exists():
            with monit.section('Build cache'):
                build_cache()

        with monit.section("Cache"):
            self.dates = np.load(str(dates_cache_path))
            self.packets = torch.tensor(np.load(str(packets_cache_path)), dtype=torch.float)

    def train_dataset(self):
        return MinutelyDataset(self.dates[:-self.validation_dates], self.packets[:-self.validation_dates])

    def valid_dataset(self):
        return MinutelyDataset(self.dates[-self.validation_dates:], self.packets[-self.validation_dates:])
