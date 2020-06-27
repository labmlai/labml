import time
from typing import Dict

import numpy as np
import requests

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

    @staticmethod
    def _parse_key(key: str):
        return key

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

        text = ['']
        step = -1
        for key, values in self.indicators.items():
            values = np.array(values)
            step = max(step, values[:, 0].max())
            text.append(f'{key} = {values[:, 1].mean()}')
        text[0] = f'step = {step}'
        text = '\n'.join(text)

        self.send(self.web_api['url'], text)

    @staticmethod
    def send(url: str, text: str):
        with requests.Session() as session:
            response = session.post(url, json={
                'text': text
            })
