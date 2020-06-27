import json
import time
import urllib.request
from typing import Dict

import numpy as np

from . import Writer as WriteBase
from ..store.indicators import Indicator
from ..store.indicators.numeric import NumericIndicator
from ...lab import lab_singleton


class Writer(WriteBase):
    def __init__(self):
        super().__init__()

        self.scalars_cache = []
        self.last_committed = time.time()
        self.web_api = lab_singleton().web_api
        self.indicators = {}
        self.run_uuid = None
        self.name = None
        self.comment = None
        self.hyperparams = None

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

    def set_hyperparams(self, hyperparams: Dict[str, any]):
        self.hyperparams = hyperparams

    def start(self):
        if self.web_api is None:
            return

        data = {
            'run_uuid': self.run_uuid,
            'name': self.name,
            'comment': self.comment,
            'params': {}
        }

        if self.hyperparams is not None:
            for k, v in self.hyperparams.items():
                data['params'][k] = v

        self.last_committed = time.time()
        self.send(self.web_api['url'], data)

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
        if self.web_api is None:
            return

        for ind in indicators.values():
            if not ind.is_print:
                continue

            self._write_indicator(global_step, ind)

        t = time.time()
        if t - self.last_committed > self.web_api['frequency']:
            self.last_committed = t
            self.flush()

    def flush(self):
        if self.web_api is None:
            return

        data = {}
        step = -1
        for key, values in self.indicators.items():
            values = np.array(values)
            step = max(step, values[:, 0].max())
            data[key] = values[:, 1].mean()
        data['step'] = step

        self.send(self.web_api['url'], {
            'run_uuid': self.run_uuid,
            'track': data
        })

    @staticmethod
    def send(url: str, data: Dict[str, any]):
        req = urllib.request.Request(url)
        req.add_header('Content-Type', 'application/json; charset=utf-8')
        data = json.dumps(data)
        data = data.encode('utf-8')
        req.add_header('Content-Length', str(len(data)))

        response = urllib.request.urlopen(req, data)
