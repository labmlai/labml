import threading
import time
from typing import Dict
from typing import Union

import numpy as np

from ..app import AppTracker, Packet, AppTrackDataSource
from ..app.url import AppUrlResponseHandler

MAX_BUFFER_SIZE = 1024

LOGS_FREQUENCY = 0


class Header(AppTrackDataSource):
    def __init__(self, app_tracker: AppTracker, *, open_browser: bool):
        super().__init__()

        self.open_browser = open_browser
        self.app_tracker = app_tracker
        self.name = None
        self.comment = None
        self.data = {}
        self.lock = threading.Lock()

    def get_data_packet(self) -> Packet:
        with self.lock:
            self.data['time'] = time.time()
            packet = Packet(self.data)
            self.data = {}
            return packet

    def start(self, configs: Dict[str, any]):
        self.app_tracker.add_handler(AppUrlResponseHandler(self.open_browser, 'Monitor computer at '))
        with self.lock:
            self.data['configs'] = configs
            self.data['name'] = 'My computer'

        self.app_tracker.has_data(self)

    def status(self, rank: int, status: str, details: str, time_: float):
        with self.lock:
            self.data['status'] = {
                'rank': rank,
                'status': status,
                'details': details,
                'time': time_
            }

        self.app_tracker.has_data(self)

        self.app_tracker.stop()


class Writer(AppTrackDataSource):
    def __init__(self, app_tracker: AppTracker, *, frequency: float):
        super().__init__()

        self.frequency = frequency
        self.app_tracker = app_tracker
        self.last_committed = time.time()
        self.data = {}
        self.lock = threading.Lock()

    @staticmethod
    def _parse_key(key: str):
        return key

    def track(self, indicators: Dict[str, Union[float, int]]):
        t = time.time()
        for k, v in indicators.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append((t, v))

        t = time.time()
        freq = self.frequency
        if t - self.last_committed > freq:
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
            if not self.data:
                return

        self.app_tracker.has_data(self)

    def get_and_clear_indicators(self):
        data = {}

        for key, value in self.data.items():
            value = np.array(value)
            timestamp: np.ndarray = value[:, 0]
            value: np.ndarray = value[:, 1]
            while value.shape[0] > MAX_BUFFER_SIZE:
                if value.shape[0] % 2 == 1:
                    value = np.concatenate((value, value[-1:]))
                    timestamp = np.concatenate((timestamp, timestamp[-1:]))

                n = value.shape[0] // 2
                timestamp = np.mean(timestamp.reshape(n, 2), axis=-1)
                value = np.mean(value.reshape(n, 2), axis=-1)

            data[key] = {
                'step': timestamp.tolist(),
                'value': value.tolist()
            }

        self.data = {}

        return data
