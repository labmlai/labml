import threading
import time
from typing import Dict

import numpy as np

from labml.internal.api import ApiCaller, Packet, ApiDataSource
from . import Writer as WriteBase
from ..indicators import Indicator
from ..indicators.numeric import NumericIndicator

MAX_BUFFER_SIZE = 1024
WARMUP_COMMITS = 5


class Writer(WriteBase, ApiDataSource):
    def __init__(self, api_caller: ApiCaller, *,
                 frequency: float):
        super().__init__()

        self.frequency = frequency
        self.api_caller = api_caller
        self.last_committed = time.time()
        self.commits_count = 0
        self.indicators = {}
        self.data = {}
        self.lock = threading.Lock()

    @staticmethod
    def _parse_key(key: str):
        return key

    def save_indicators(self, dot_indicators: Dict[str, Indicator], indicators: Dict[str, Indicator]):
        wildcards = {k: ind.to_dict() for k, ind in dot_indicators.items()}
        inds = {k: ind.to_dict() for k, ind in indicators.items()}
        with self.lock:
            self.data['wildcard_indicators'] = wildcards
            self.data['indicators'] = inds
        self.api_caller.has_data(self)

    def _write_indicator(self, global_step: int, indicator: Indicator):
        if indicator.is_empty():
            return

        if isinstance(indicator, NumericIndicator):
            value = indicator.get_mean()
            key = self._parse_key(indicator.mean_key)
            if key not in self.indicators:
                self.indicators[key] = []

            self.indicators[key].append((global_step, value))

    def write(self, *,
              global_step: int,
              indicators: Dict[str, Indicator]):
        for ind in indicators.values():
            self._write_indicator(global_step, ind)

        t = time.time()
        freq = self.frequency
        if self.commits_count < WARMUP_COMMITS:
            freq /= 2 ** (WARMUP_COMMITS - self.commits_count)
        if t - self.last_committed > freq:
            self.commits_count += 1
            self.flush()

    def get_data_packet(self) -> Packet:
        with self.lock:
            self.last_committed = time.time()
            self.data['track'] = self.get_and_clear_indicators()
            self.data['time'] = time.time()
            packet = Packet(self.data)
            self.data = {}
            return packet

    def flush(self):
        with self.lock:
            if not self.indicators:
                return

        self.api_caller.has_data(self)

    def get_and_clear_indicators(self):
        data = {}

        indicators = self.indicators
        self.indicators = {}

        for key, value in indicators.items():
            value = np.array(value)
            step: np.ndarray = value[:, 0]
            value: np.ndarray = value[:, 1]
            while value.shape[0] > MAX_BUFFER_SIZE:
                if value.shape[0] % 2 == 1:
                    value = np.concatenate((value, value[-1:]))
                    step = np.concatenate((step, step[-1:]))

                n = value.shape[0] // 2
                step = np.mean(step.reshape(n, 2), axis=-1)
                value = np.mean(value.reshape(n, 2), axis=-1)

            data[key] = {
                'step': step.tolist(),
                'value': value.tolist()
            }

        return data
