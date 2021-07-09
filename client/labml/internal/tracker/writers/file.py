import json
import threading
import time
from pathlib import PurePath
from queue import Queue
from typing import Dict, Optional

import numpy as np

from . import Writer as WriteBase
from ..indicators import Indicator
from ..indicators.numeric import NumericIndicator

MAX_BUFFER_SIZE = 1024
WARMUP_COMMITS = 5
TIMEOUT_SECONDS = 5
FREQUENCY = 60


class FileWriterThread(threading.Thread):
    def __init__(self, file_path: PurePath):
        super().__init__(daemon=False)
        self.file_path = file_path
        self.queue = Queue()

    def push(self, data: any):
        self.queue.put(data)

    def run(self):
        while True:
            data = self.queue.get()
            if data == 'done':
                return
            else:
                self._process(data)

    def _process(self, data: Dict[str, any]):
        res = self._send(data['packet'])
        if 'accept' in data:
            data['accept'](res)

    def _send(self, data: Dict[str, any]):
        with open(str(self.file_path), 'a') as f:
            f.write(json.dumps(data) + '\n')


class Writer(WriteBase):
    url: Optional[str]
    thread: Optional[FileWriterThread]

    def __init__(self, file_path: PurePath):
        super().__init__()

        self.file_path = file_path
        self.scalars_cache = []
        self.last_committed = time.time()
        self.commits_count = 0
        self.indicators = {}

        self.thread = FileWriterThread(file_path)

    @staticmethod
    def _parse_key(key: str):
        return key

    def _write_indicator(self, global_step: int, indicator: Indicator):
        if indicator.is_empty():
            return

        if isinstance(indicator, NumericIndicator):
            mean_value = indicator.get_mean()
            hist = indicator.get_histogram()
            if hist:
                if len(hist) > 1:
                    hist, bins = np.histogram(hist)
                    hist = {'hist': hist.tolist(), 'bins': bins.tolist()}
                else:
                    hist = None
            key = self._parse_key(indicator.mean_key)
            if key not in self.indicators:
                self.indicators[key] = {
                    'mean': [],
                    'hist': []
                }

            self.indicators[key]['mean'].append((global_step, mean_value))
            self.indicators[key]['hist'].append(hist)

    def write(self, *,
              global_step: int,
              indicators: Dict[str, Indicator]):
        for ind in indicators.values():
            # if not ind.is_print:
            #     continue
            self._write_indicator(global_step, ind)

        t = time.time()
        freq = FREQUENCY
        if self.commits_count < WARMUP_COMMITS:
            freq /= 2 ** (WARMUP_COMMITS - self.commits_count)
        if t - self.last_committed > freq:
            self.last_committed = t
            self.commits_count += 1
            self.flush()

    def flush(self):
        data = {}

        for key, ind_values in self.indicators.items():
            value = np.array(ind_values['mean'])
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
                'value': value.tolist(),
                'hist': ind_values['hist']
            }

        self.indicators = {}

        self.thread.push({'packet': {
            'track': data,
            'time': time.time()
        }})
        if not self.thread.is_alive():
            self.thread.start()
