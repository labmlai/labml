import sys
import time
from io import StringIO
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from labml.internal.api import ApiCaller

WARMUP_COMMITS = 5


class ApiLogs:
    api_caller: Optional['ApiCaller']
    frequency: float

    def __init__(self):
        super().__init__()

        self.api_caller = None
        self.frequency = 1
        self.last_committed = time.time()
        self.commits_count = 0
        self.stdout = ''
        self.stderr = ''
        self.logger = ''

    def set_api(self, api_caller: 'ApiCaller', *, frequency: float):
        self.api_caller = api_caller
        self.frequency = frequency
        self.check_and_flush()

    def check_and_flush(self):
        if self.api_caller is None:
            return
        if self.stdout == '' and self.stderr == '' and self.logger == '':
            return
        t = time.time()
        freq = self.frequency
        if self.commits_count < WARMUP_COMMITS:
            freq /= 2 ** (WARMUP_COMMITS - self.commits_count)
        if self.stderr != '' or self.commits_count == 0 or t - self.last_committed > freq:
            self.last_committed = t
            self.commits_count += 1
            self.flush()

    def outputs(self, *,
                stdout_: str = '',
                stderr_: str = '',
                logger_: str = ''):
        if stdout_ != '':
            self.stdout += stdout_
        if stderr_ != '':
            self.stderr += stderr_
        if logger_ != '':
            self.logger += logger_

        self.check_and_flush()

    def flush(self):
        data = {'time': time.time()}
        if self.stdout != '':
            data['stdout'] = self.stdout
            self.stdout = ''
        if self.stderr != '':
            data['stderr'] = self.stderr
            self.stderr = ''
        if self.logger != '':
            data['logger'] = self.logger
            self.logger = ''

        from labml.internal.api import Packet
        self.api_caller.push(Packet(data))


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


def capture():
    sys.stderr = OutputStream(sys.stderr, 'stderr_')
    sys.stdout = OutputStream(sys.stdout, 'stdout_')


capture()
