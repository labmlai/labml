import time
import webbrowser
from typing import Dict, Optional

import numpy as np

from labml import logger
from labml.internal.api import ApiCaller, Packet
from labml.logger import Text
from . import Writer as WriteBase
from ..indicators import Indicator
from ..indicators.numeric import NumericIndicator
from ...configs.processor import ConfigsSaver

MAX_BUFFER_SIZE = 1024
WARMUP_COMMITS = 5

STATIC_ATTRIBUTES = {'configs', 'wildcard_indicators', 'indicators'}


class WebApiConfigsSaver(ConfigsSaver):
    def __init__(self, writer: 'Writer'):
        self.writer = writer

    def save(self, configs: Dict):
        self.writer.save_configs(configs)


class Writer(WriteBase):
    configs_saver: Optional[WebApiConfigsSaver]
    url: Optional[str]

    def __init__(self, api_caller: ApiCaller, *,
                 frequency: float,
                 open_browser: bool):
        super().__init__()

        self.open_browser = open_browser
        self.frequency = frequency
        self.api_caller = api_caller
        self.scalars_cache = []
        self.last_committed = time.time()
        self.commits_count = 0
        self.indicators = {}
        self.run_uuid = None
        self.name = None
        self.comment = None
        self.configs = None
        self.state = None
        self.configs_saver = None

    @staticmethod
    def _parse_key(key: str):
        return key

    def set_info(self, *,
                 run_uuid: str,
                 name: str,
                 comment: str):
        self.run_uuid = run_uuid
        self.name = name
        self.comment = comment

    def get_configs_saver(self):
        if self.configs_saver is None:
            self.configs_saver = WebApiConfigsSaver(self)
        return self.configs_saver

    def save_configs(self, configs: Dict[str, any]):
        self.api_caller.push(Packet({'configs': configs}))

    def start(self):
        data = {
            'name': self.name,
            'comment': self.comment,
            'time': time.time(),
            'configs': {}
        }

        self.last_committed = time.time()
        self.commits_count = 0

        self.api_caller.push(Packet(data, callback=self._started))

    def save_indicators(self, dot_indicators: Dict[str, Indicator], indicators: Dict[str, Indicator]):
        wildcards = {k: ind.to_dict() for k, ind in dot_indicators.items()}
        inds = {k: ind.to_dict() for k, ind in indicators.items()}

        self.api_caller.push(Packet({'wildcard_indicators': wildcards, 'indicators': inds}))

    def _started(self, url):
        if url is None:
            return None

        logger.log([('Monitor experiment at ', Text.meta), (url, Text.link)])
        if self.open_browser:
            webbrowser.open(url)

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
            self.last_committed = t
            self.commits_count += 1
            self.flush()

    def status(self, rank: int, status: str, details: str, time_: float):
        self.state = {
            'rank': rank,
            'status': status,
            'details': details,
            'time': time_
        }

        # TODO: Will have to fix this when there are other statuses that dont stop the experiment
        # This will stop the thread after sending all the data
        self.api_caller.stop()

    def flush(self):
        data = {}

        for key, value in self.indicators.items():
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

        self.indicators = {}

        self.api_caller.push(Packet({
            'track': data,
            'status': self.state,
            'time': time.time()
        }))
