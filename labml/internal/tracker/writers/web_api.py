import json
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from queue import Queue
from typing import Dict, Optional

import numpy as np

import labml
from labml import logger
from labml.internal.lab import lab_singleton
from labml.logger import Text
from labml.utils.notice import labml_notice
from . import Writer as WriteBase
from ..indicators import Indicator
from ..indicators.numeric import NumericIndicator
from ...configs.processor import ConfigsSaver

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
        self.warned_updating = False
        self.error = False

    def push(self, data: any):
        self.queue.put(data)

    def stop(self):
        self.is_stopped = True
        self.push('done')

    def run(self):
        while True:
            data = self.queue.get()
            if data == 'done':
                if self.warned_updating:
                    logger.log('Finished updating LabML App.', Text.highlight)
                return
            else:
                if self.is_stopped and not self.warned_updating:
                    logger.log('Still updating LabML App, please wait for it to complete..', Text.highlight)
                    self.warned_updating = True
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

            if result['errors']:
                for e in result['errors']:
                    if 'error' in e:
                        labml_notice(['LabML App Error: ', (e['error'] + '', Text.key), '\n',
                                      (e['message'], Text.value),
                                      f'App URL: {self.url}'],
                                     is_danger=True)
                        self.error = True
                        raise RuntimeError('LabML App Error', e)
                    elif 'warning' in e:
                        labml_notice(
                            ['LabML App Warning: ', (e['warning'] + ': ', Text.key), (e['message'], Text.value)])
                    else:
                        self.error = True
                        raise RuntimeError('Unknown error from LabML App', e)
            return result.get('url', None)
        except urllib.error.HTTPError as e:
            labml_notice([f'Failed to send to {self.url}: ',
                          (str(e.code), Text.value),
                          '\n' + str(e.reason)])
            return None
        except urllib.error.URLError as e:
            labml_notice([f'Failed to connect to {self.url}\n',
                          str(e.reason)])
            return None
        except socket.timeout as e:
            labml_notice([f'{self.url} timeout\n',
                          str(e)])
            return None
        except ConnectionResetError as e:
            labml_notice([f'Connection reset by LabML App server {self.url}\n',
                          str(e)])
            return None


class WebApiConfigsSaver(ConfigsSaver):
    def __init__(self, writer: 'Writer'):
        self.writer = writer

    def save(self, configs: Dict):
        self.writer.save_configs(configs)


class Writer(WriteBase):
    configs_saver: Optional[WebApiConfigsSaver]
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
        self.thread = None
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

    def push(self, data):
        if self.thread is None:
            url = f'{self.web_api.url}run_uuid={self.run_uuid}&labml_version={labml.__version__}'
            self.thread = WebApiThread(url, self.web_api.verify_connection)

        self.thread.push(data)
        if not self.thread.is_alive():
            self.thread.start()

    def get_configs_saver(self):
        if self.configs_saver is None:
            self.configs_saver = WebApiConfigsSaver(self)
        return self.configs_saver

    def save_configs(self, configs: Dict[str, any]):
        if self.web_api is None:
            return

        data = {'configs': configs}

        self.push({'packet': data})

    def start(self):
        if self.web_api is None:
            return

        data = {
            'name': self.name,
            'comment': self.comment,
            'time': time.time(),
            'configs': {}
        }

        self.last_committed = time.time()
        self.commits_count = 0

        self.push({'packet': data, 'accept': self._started})

    def save_indicators(self, dot_indicators: Dict[str, Indicator], indicators: Dict[str, Indicator]):
        if self.web_api is None:
            return

        wildcards = {k: ind.to_dict() for k, ind in dot_indicators.items()}
        inds = {k: ind.to_dict() for k, ind in indicators.items()}
        data = {
            'wildcard_indicators': wildcards,
            'indicators': inds
        }

        self.push({'packet': data})

    def _started(self, url):
        if url is None:
            return None

        logger.log([('Monitor experiment at ', Text.meta), (url, Text.link)])
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

        if self.thread.error:
            raise RuntimeError('LabML App error: See above for error details')

        for ind in indicators.values():
            self._write_indicator(global_step, ind)

        t = time.time()
        freq = self.web_api.frequency
        if self.commits_count < WARMUP_COMMITS:
            freq /= 2 ** (WARMUP_COMMITS - self.commits_count)
        if t - self.last_committed > freq:
            self.last_committed = t
            self.commits_count += 1
            self.flush()

    def status(self, rank: int, status: str, details: str, time_: float):
        if self.web_api is None:
            return

        self.state = {
            'rank': rank,
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

        self.push({'packet': {
            'track': data,
            'status': self.state,
            'time': time.time()
        }})
