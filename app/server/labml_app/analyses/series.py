import base64
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

import numpy as np

MAX_BUFFER_LENGTH = 1024
OUTLIER_MARGIN = 0.04

SeriesModel = Dict[str, Union[np.ndarray, List[float], float]]


def _remove_old(values, steps, last_steps):
    break_time = datetime.fromtimestamp(steps[-1]) - timedelta(days=1)

    left = 0
    right = len(steps) - 1
    while left < right:
        m = (left + right) // 2
        if datetime.fromtimestamp(steps[m]) < break_time:
            left = m + 1
        else:
            right = m

    if datetime.fromtimestamp(steps[left]) < break_time:
        return values, steps

    # find index to break
    break_index = left

    return values[break_index:], steps[break_index:], last_steps[break_index:]


class Series:
    step: np.ndarray
    last_step: np.ndarray
    value: np.ndarray
    step_gap: float
    max_buffer_length: int
    keep_last_24h: bool

    def __init__(self, max_buffer_length: int = None, keep_last_24h: bool = False):
        self.step = np.array([])
        self.last_step = np.array([])
        self.value = np.array([])
        self.step_gap = 0
        self.keep_last_24h = keep_last_24h
        if max_buffer_length:
            self.max_buffer_length = max_buffer_length
        else:
            self.max_buffer_length = MAX_BUFFER_LENGTH

        try:
            import labml_fast_merge
            self.labml_fast_merge = labml_fast_merge

            self._merge = self._merge_new
        except ImportError:
            self._merge = self._merge_old

    @property
    def last_value(self) -> float:
        return self.value[-1]

    def find_step_gap(self):
        if len(self) > 1:
            return max(1., (self.last_step[1] - self.last_step[0]).item())
        else:
            return 1.

    def _data_to_binary(self):
        values = base64.b64encode(np.array(self.value, dtype=np.float32).tobytes()).decode('utf-8')
        steps = base64.b64encode(np.array(self.step, dtype=np.float32).tobytes()).decode('utf-8')
        last_steps = base64.b64encode(np.array(self.last_step, dtype=np.float32).tobytes()).decode('utf-8')

        return values, steps, last_steps

    @property
    def detail(self) -> SeriesModel:
        values, steps, last_steps = self._data_to_binary()
        return {
            'step': steps,
            'value': values,
            'mean': np.mean(self.value),
            'last_step': last_steps,
        }

    @property
    def summary(self) -> Dict[str, np.ndarray]:
        return {
            'mean': np.mean(self.value)
        }

    def to_data(self) -> SeriesModel:
        return {
            'step': self.step,
            'value': self.value,
            'last_step': self.last_step,
            'step_gap': self.step_gap,
            'mean': np.mean(self.value),
        }

    def __len__(self):
        return len(self.last_step)

    def update(self, step: List[float], value: List[float]) -> None:
        prev_size = len(self.value)

        value = np.array(value)
        step = np.array(step)
        last_step = np.array(step)

        self._remove_nan(value)

        self.value = np.concatenate((self.value, value))
        self.step = np.concatenate((self.step, step))
        self.last_step = np.concatenate((self.last_step, last_step))

        if self.keep_last_24h:
            self.value, self.step, self.last_step = _remove_old(self.value, self.step, self.last_step)

        self.step_gap = self.find_step_gap()

        # with monit.section('Merge added'):
        self.merge(prev_size)

        while len(self) > self.max_buffer_length:
            self.step_gap *= 2
            # with monit.section('Merge'):
            self.merge()

    def _remove_nan(self, values) -> None:
        infin = np.isfinite(values)
        np.bitwise_not(infin, out=infin)

        if infin[0]:
            values[0] = 0.0 if len(self.value) == 0 else self.value[-1]
        for i in range(1, len(values)):
            if infin[i]:
                values[i] = values[i - 1]

    def _merge_new(self,
                   values: np.ndarray,
                   last_step: np.ndarray,
                   steps: np.ndarray,
                   prev_last_step: int = 0,
                   i: int = 0):  # from_step

        return self.labml_fast_merge.merge(values, last_step, steps, float(self.step_gap), float(prev_last_step), i)

    def _merge_old(self,
                   values: np.ndarray,
                   last_step: np.ndarray,
                   steps: np.ndarray,
                   prev_last_step: int = 0,
                   i: int = 0  # from_step
                   ):
        # Assumes that steps are evenly spaced
        j = i + 1
        while j < len(values):
            if last_step[j] - prev_last_step < self.step_gap or last_step[j] - last_step[j - 1] < 1e-3:  # merge
                iw = max(1., last_step[i] - prev_last_step)
                jw = max(1., last_step[j] - last_step[i])
                steps[i] = (steps[i] * iw + steps[j] * jw) / (iw + jw)
                values[i] = (values[i] * iw + values[j] * jw) / (iw + jw)
                last_step[i] = last_step[j]
                j += 1
            else:  # move to next
                prev_last_step = last_step[i]
                i += 1
                last_step[i] = last_step[j]
                steps[i] = steps[j]
                values[i] = values[j]
                j += 1

        return i + 1  # size after merging

    def merge(self, prev_size: int = 0):
        from_step = max(0, prev_size - 1)
        if len(self) - from_step <= 1:
            return

        if from_step > 0:
            prev_last_step = self.last_step[from_step - 1].item()
        else:
            prev_last_step = 0

        n = self._merge(self.value, self.last_step, self.step, prev_last_step, from_step)

        self.last_step = self.last_step[:n]
        self.step = self.step[:n]
        self.value = self.value[:n]

    def get_extent(self, is_remove_outliers: bool):
        if len(self.value) == 0:
            return [0, 0]
        elif len(self.value) < 10:
            return [min(self.value), max(self.value)]
        elif not is_remove_outliers:
            return [min(self.value), max(self.value)]

        values = np.sort(self.value)
        margin = max(int(len(values) * OUTLIER_MARGIN), 1)
        std_dev = np.std(self.value[margin:-margin])
        start = 0
        while start < margin:
            if values[start] + std_dev * 2 > values[margin]:
                break
            start += 1
        end = len(values) - 1
        while end > len(values) - margin - 1:
            if values[end] - std_dev * 2 < values[-margin]:
                break
            end -= 1

        return [values[start], values[end]]

    def load(self, data):
        self.step = data['step'].copy()
        self.last_step = data['last_step'].copy()
        self.value = data['value'].copy()

        return self
