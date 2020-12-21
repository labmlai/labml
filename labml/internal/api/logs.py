import sys
import threading
import time
from io import StringIO
from typing import Optional

from labml.internal.api import ApiCaller, ApiDataSource, Packet

WARMUP_COMMITS = 5


class ApiLogs(ApiDataSource):
    api_caller: Optional[ApiCaller]
    frequency: float

    def __init__(self):
        super().__init__()

        self.api_caller = None
        self.frequency = 1
        self.last_committed = time.time()
        self.commits_count = 0
        self.data = {}
        self.lock = threading.Lock()

    def set_api(self, api_caller: ApiCaller, *, frequency: float):
        self.api_caller = api_caller
        self.frequency = frequency
        self.check_and_flush()

    def check_and_flush(self):
        if self.api_caller is None:
            return
        with self.lock:
            if not self.data:
                return
        t = time.time()
        freq = self.frequency
        if self.commits_count < WARMUP_COMMITS:
            freq /= 2 ** (WARMUP_COMMITS - self.commits_count)
        if self.data.get('stderr', '') != '' or self.commits_count == 0 or t - self.last_committed > freq:
            self.commits_count += 1
            self.api_caller.has_data(self)

    def get_data_packet(self) -> Packet:
        with self.lock:
            self.last_committed = time.time()
            self.data['time'] = time.time()
            packet = Packet(self.data)
            self.data = {}
            return packet

    def outputs(self, *,
                stdout_: str = '',
                stderr_: str = '',
                logger_: str = ''):
        with self.lock:
            if stdout_ != '':
                self.data['stdout'] = self.data.get('stdout', '') + stdout_
            if stderr_ != '':
                self.data['stderr'] = self.data.get('stderr', '') + stderr_
            if logger_ != '':
                self.data['logger'] = self.data.get('logger', '') + logger_

        self.check_and_flush()


API_LOGS = ApiLogs()


class OutputStream(StringIO):
    def write(self, *args, **kwargs):  # real signature unknown
        super().write(*args, **kwargs)
        save = StringIO()
        save.write(*args, **kwargs)
        API_LOGS.outputs(**{self.type_: save.getvalue()})
        self.original.write(*args, **kwargs)

    def __init__(self, original, type_):  # real signature unknown
        super().__init__()
        self.type_ = type_
        self.original = original


_original_stdout_write = sys.stdout.write
_original_stderr_write = sys.stderr.write


def _write_stdout(*args, **kwargs):
    _original_stdout_write(*args, **kwargs)
    save = StringIO()
    save.write(*args, **kwargs)
    API_LOGS.outputs(stdout_=save.getvalue())


def _write_stderr(*args, **kwargs):
    _original_stderr_write(*args, **kwargs)
    save = StringIO()
    save.write(*args, **kwargs)
    API_LOGS.outputs(stderr_=save.getvalue())


def capture():
    sys.stdout.write = _write_stdout
    sys.stderr.write = _write_stderr


capture()
