import json
import threading
import time
from pathlib import PurePath
from queue import Queue
from typing import Dict, Optional

from . import Writer as WriteBase
from ..indicators import Indicator
from ..indicators.numeric import NumericIndicator
from ...util.values import to_numpy

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
        self._send(data['packet'])

    def _send(self, data: Dict[str, any]):
        with open(str(self.file_path), 'a') as f:
            f.write(json.dumps(data) + '\n')


def _to_list(value):
    return to_numpy(value).tolist()


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
        self.is_thread_started = False

    @staticmethod
    def _parse_key(key: str):
        return key

    def _write_indicator(self, global_step: int, indicator: Indicator):
        if indicator.is_empty():
            return

        if isinstance(indicator, NumericIndicator):
            values = indicator.get_all_values()
            key = self._parse_key(indicator.mean_key)
            if key not in self.indicators:
                self.indicators[key] = []

            self.indicators[key].append((global_step, _to_list(values)))

    def write(self, *,
              global_step: int,
              indicators: Dict[str, Indicator]):
        for ind in indicators.values():
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
            data[key] = ind_values

        self.indicators = {}

        self.thread.push({'packet': {
            'track': data,
            'time': time.time()
        }})
        if not self.is_thread_started:
            self.is_thread_started = True
            self.thread.start()

    def finish(self):
        self.flush()
        self.thread.push('done')
