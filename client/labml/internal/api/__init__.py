import json
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from queue import Queue
from typing import Dict, Optional, Set, List

import labml
from labml import logger
from labml.logger import Text
from labml.utils.notice import labml_notice

UPDATING_APP_MESSAGE = 'Updating App. Please wait'


@dataclass
class Packet:
    data: Dict[str, any]
    idx: int = -1


class ApiDataSource:
    def get_data_packet(self) -> Packet:
        raise NotImplementedError


class SimpleApiDataSource(ApiDataSource):
    def __init__(self, data: Dict[str, any]):
        self.data = data

    def get_data_packet(self) -> Packet:
        return Packet(data=self.data)


class ApiResponseHandler:
    def handle(self, data) -> bool:
        raise NotImplementedError


class _WebApiThread(threading.Thread):
    def __init__(self, url: str, *, timeout_seconds: int, daemon: bool):
        super().__init__(daemon=daemon)
        self.please_wait_count = 0
        self.timeout_seconds = timeout_seconds
        self.url = url
        self.queue: Queue[ApiDataSource] = Queue()
        self.is_stopped = False
        self.errored = False
        self.handlers: List[ApiResponseHandler] = []

    def push_data_source(self, data_source: ApiDataSource):
        self.queue.put(data_source)

    def stop(self):
        self.is_stopped = True
        logger.log('Still updating app.labml.ai, please wait for it to complete...', Text.highlight)
        self.please_wait_count = 1

    def add_handler(self, handler: ApiResponseHandler):
        self.handlers.append(handler)

    @staticmethod
    def _is_updating_notification(packet: Packet):
        """
        Checks if we just printed app updating notification.
        If so filter those out.
        """
        data = packet.data
        if set(data.keys()) != {'stdout', 'time'} and set(data.keys()) != {'logger', 'time'}:
            return False

        if 'stdout' in data:
            lines = data['stdout'].split('\n')
        else:
            lines = data['logger'].split('\n')

        if len(lines) == 1 and lines[0].find(UPDATING_APP_MESSAGE) != -1:
            return True

        return False

    def _get_packets(self) -> List[Packet]:
        sources = [self.queue.get()]
        while not self.queue.empty():
            sources.append(self.queue.get())

        sources = list(set(sources))

        packets = [s.get_data_packet() for s in sources]
        return [p for p in packets if not self._is_updating_notification(p)]

    def run(self):
        while True:
            packets = self._get_packets()
            if self.is_stopped:
                if not packets:
                    logger.log()
                    logger.log('Finished updating LabML App.', Text.highlight)
                    return

                logger.log(UPDATING_APP_MESSAGE + '...' * self.please_wait_count, Text.meta,
                           is_new_line=False)
                self.please_wait_count += 1
            retries = 0
            while True:
                if self._process(packets):
                    break
                else:
                    logger.log(f'Retrying again in 10 seconds ({retries})...', Text.highlight)
                    time.sleep(10)
                    retries += 1
            if self.is_stopped and self.queue.empty():
                time.sleep(0.5)
                if self.queue.empty():
                    logger.log()
                    logger.log('Finished updating LabML App.', Text.highlight)
                    return

    def _process(self, packets: List[Packet]) -> bool:
        if not packets:
            return True

        data = [p.data for p in packets]

        try:
            response = self._send(data)
        except urllib.error.HTTPError as e:
            labml_notice([f'Failed to send to {self.url}: ',
                          (str(e.code), Text.value),
                          '\n' + str(e.reason)])
            return False
        except urllib.error.URLError as e:
            labml_notice([f'Failed to connect to {self.url}\n',
                          str(e.reason)])
            return False
        except socket.timeout as e:
            labml_notice([f'{self.url} timeout\n',
                          str(e)])
            return False
        except ConnectionResetError as e:
            labml_notice([f'Connection reset by LabML App server {self.url}\n',
                          str(e)])
            return False

        for h in self.handlers:
            if h.handle(response):
                break

        return True

    def _send(self, data: List[Dict[str, any]]) -> Dict:
        req = urllib.request.Request(self.url)
        req.add_header('Content-Type', 'application/json; charset=utf-8')
        data_json = json.dumps(data)
        data_json = data_json.encode('utf-8')
        req.add_header('Content-Length', str(len(data_json)))

        # print('Data size', len(data_json))
        response = urllib.request.urlopen(req, data_json, timeout=self.timeout_seconds)
        content = response.read().decode('utf-8')
        result = json.loads(content)

        for e in result.get('errors', []):
            if 'error' in e:
                labml_notice(['LabML App Error: ', (e['error'] + '', Text.key), '\n',
                              (e['message'], Text.value)],
                             is_danger=True)
                self.errored = True
                raise RuntimeError('LabML App Error', e)
            elif 'warning' in e:
                labml_notice(
                    ['LabML App Warning: ', (e['warning'] + ': ', Text.key), (e['message'], Text.value)])
            else:
                self.errored = True
                raise RuntimeError('Unknown error from LabML App', e)

        return result


class ApiCaller:
    web_api_url: str
    params: Dict[str, str]
    thread: Optional[_WebApiThread]
    state_attributes: Set[str]

    def __init__(self, web_api_url: str, params: Dict[str, str], *,
                 timeout_seconds: int = 15,
                 daemon: bool = False):
        super().__init__()

        self.daemon = daemon
        params.copy()
        params['labml_version'] = labml.__version__
        params = '&'.join([f'{k}={v}' for k, v in params.items()])

        self.web_api_url = f'{web_api_url}{params}'
        self.timeout_seconds = timeout_seconds
        self.thread = None
        self.stopped = False

    def _check(self) -> bool:
        if self.stopped:
            return False

        if self.thread is None:
            self.thread = _WebApiThread(self.web_api_url,
                                        timeout_seconds=self.timeout_seconds,
                                        daemon=self.daemon)

        if self.thread.errored:
            raise RuntimeError('LabML App error: See above for error details')

        return True

    def has_data(self, source: ApiDataSource):
        if not self._check():
            return False

        self.thread.push_data_source(source)
        if not self.thread.is_alive():
            self.thread.start()

    def add_handler(self, handler: ApiResponseHandler):
        if not self._check():
            return False

        self.thread.add_handler(handler)

    def stop(self):
        self.stopped = True
        self.thread.stop()
