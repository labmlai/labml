import json
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
import warnings
import webbrowser
from queue import Queue
from typing import Dict, Optional

import numpy as np

from labml import logger
from labml.logger import Text
from . import Writer as WriteBase
from ..store.indicators import Indicator
from ..store.indicators.numeric import NumericIndicator
from ...lab import lab_singleton

MAX_BUFFER_SIZE = 1024
WARMUP_COMMITS = 5
TIMEOUT_SECONDS = 5


class WebApiThread(threading.Thread):
    def __init__(self, url: str, verify_connection: bool):
        super().__init__(daemon=False)
        self.verify_connection = verify_connection
        self.url = url
        self.queue = Queue()
        self.is_stopped = False

    def push(self, data: any):
        self.queue.put(data)

    def stop(self):
        self.is_stopped = True
        self.push('done')

    def run(self):
        while True:
            data = self.queue.get()
            if data == 'done':
                return
            else:
                if self.is_stopped:
                    print('Updating web api...')
                self._process(data)

    def _process(self, data: Dict[str, any]):
        res = self._send(data['packet'])
        if 'accept' in data:
            data['accept'](res)

    def _send(self, data: Dict[str, any]):
        req = urllib.request.Request(self.url)
        req.add_header('Content-Type', 'application/json; charset=utf-8')
        data = json.dumps(data)
        data = data.encode('utf-8')
        req.add_header('Content-Length', str(len(data)))

        try:
            if self.verify_connection:
                response = urllib.request.urlopen(req, data, timeout=TIMEOUT_SECONDS)
            else:
                response = urllib.request.urlopen(req, data, timeout=TIMEOUT_SECONDS,
                                                  context=ssl._create_unverified_context())
            content = response.read().decode('utf-8')
            result = json.loads(content)
            for e in result['errors']:
                warnings.warn(f"WEB API error {e['error']} : {e['message']}")
            return result.get('url', None)
        except urllib.error.HTTPError as e:
            warnings.warn(f"Failed to send message to WEB API  {self.url}: {e}")
            return None
        except urllib.error.URLError as e:
            warnings.warn(f"Failed to connect to WEB API {self.url}: {e}")
            return None
        except socket.timeout as e:
            warnings.warn(f"WEB API timeout {self.url}: {e}")
            return None


class Writer(WriteBase):
    url: Optional[str]
    thread: Optional[WebApiThread]

    def __init__(self):
        super().__init__()

        self.scalars_cache = []
        self.last_committed = time.time()
        self.commits_count = 0
        self.indicators = {}
        self.run_uuid = None
        self.name = None
        self.comment = None
        self.configs = None
        self.web_api = lab_singleton().web_api
        self.state = None

        if self.web_api is not None:
            self.thread = WebApiThread(self.web_api.url, self.web_api.verify_connection)
        else:
            self.thread = None

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

    def set_configs(self, configs: Dict[str, any]):
        self.configs = configs

    def start(self):
        if self.web_api is None:
            return

        data = {
            'run_uuid': self.run_uuid,
            'name': self.name,
            'comment': self.comment,
            'time': time.time(),
            'configs': {}
        }

        if self.configs is not None:
            for k, v in self.configs.items():
                data['configs'][k] = v

        self.last_committed = time.time()
        self.commits_count = 0
        self.thread.push({'packet': data,
                          'accept': self._started})
        if not self.thread.is_alive():
            self.thread.start()

    def save_indicators(self, dot_indicators: Dict[str, Indicator], indicators: Dict[str, Indicator]):
        if self.web_api is None:
            return

        wildcards = {k: ind.to_dict() for k, ind in dot_indicators.items()}
        inds = {k: ind.to_dict() for k, ind in indicators.items()}
        data = {
            'run_uuid': self.run_uuid,
            'wildcard_indicators': wildcards,
            'indicators': inds
        }

        self.thread.push({'packet': data})

    def _started(self, url):
        if url is None:
            return None

        logger.log([('Monitor experiment at ', Text.meta), (url, Text.highlight)])
        if self.web_api.open_browser:
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
        if self.web_api is None:
            return

        for ind in indicators.values():
            if not ind.is_print:
                continue

            self._write_indicator(global_step, ind)

        t = time.time()
        freq = self.web_api.frequency
        if self.commits_count < WARMUP_COMMITS:
            freq /= 2 ** (WARMUP_COMMITS - self.commits_count)
        if t - self.last_committed > freq:
            self.last_committed = t
            self.commits_count += 1
            self.flush()

    def status(self, status: str, details: str, time_: float):
        self.state = {
            'status': status,
            'details': details,
            'time': time_
        }

        self.flush()

        # TODO: Will have to fix this when there are other statuses that dont stop the experiment
        self.thread.stop()

    def flush(self):
        if self.web_api is None:
            return

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

        self.thread.push({'packet': {
            'run_uuid': self.run_uuid,
            'track': data,
            'status': self.state,
            'time': time.time()
        }})
