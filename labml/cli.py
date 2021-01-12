import argparse
import subprocess
import threading
import time
import webbrowser
from typing import List

from typing.io import IO

from labml import logger, experiment
from labml.experiment import generate_uuid
from labml.internal.api import ApiCaller, SimpleApiDataSource
from labml.internal.api.logs import ApiLogs
from labml.logger import Text


def _open_dashboard():
    try:
        import labml_dashboard
    except (ImportError, ModuleNotFoundError):
        logger.log("Cannot import ", ('labml_dashboard', Text.highlight), '.')
        logger.log('Install with ',
                   ('pip install labml_dashboard', Text.value))
        return

    labml_dashboard.start_server()


class ExecutorThread(threading.Thread):
    process: subprocess.Popen

    def __init__(self, command: str, api_logs: ApiLogs):
        super().__init__(daemon=False)
        self.api_logs = api_logs
        self.command = command
        self.exit_code = 0

    def _read(self, stream: IO, name: str):
        while stream.readable():
            data = stream.read(1024).decode('utf-8')
            if len(data) == 0:
                break
            print(data)
            self.api_logs.outputs(**{name: data})

    def _read_stdout(self):
        self._read(self.process.stdout, 'stdout_')

    def _read_stderr(self):
        self._read(self.process.stderr, 'stderr_')

    def run(self):
        self.process = subprocess.Popen(self.command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        while True:
            self._read_stdout()
            self._read_stderr()
            if self.process.poll() is not None:
                break
            time.sleep(0.1)
        self.exit_code = self.process.returncode

        self._read_stdout()
        self._read_stderr()


def _capture(args: List[str]):
    api_caller = ApiCaller("https://api.lab-ml.com/api/v1/track?", {'run_uuid': generate_uuid()})
    api_logs = ApiLogs()
    data = {
        'name': 'Capture',
        'comment': ' '.join(args),
        'time': time.time()
    }

    def _started(url):
        if url is None:
            return None

        logger.log([('Monitor experiment at ', Text.meta), (url, Text.link)])
        webbrowser.open(url)

    api_caller.has_data(SimpleApiDataSource(data, callback=_started))
    api_logs.set_api(api_caller, frequency=0)

    thread = ExecutorThread(' '.join(args), api_logs)
    thread.start()
    thread.join()
    data = {
        'rank': 0,
        'status': 'completed',
        'details': None,
        'time': time.time()
    }

    api_caller.has_data(SimpleApiDataSource({
        'status': data,
        'time': time.time()
    }))

    api_caller.stop()


def _launch(args: List[str]):
    import sys
    import os

    if 'RUN_UUID' not in os.environ:
        os.environ['RUN_UUID'] = experiment.generate_uuid()

    cwd = os.getcwd()
    if 'PYTHONPATH' in os.environ:
        python_path = os.environ['PYTHONPATH']
        print(python_path)
        os.environ['PYTHONPATH'] = f"{python_path}:{cwd}:{cwd}/src"
    else:
        os.environ['PYTHONPATH'] = f"{cwd}:{cwd}/src"

    cmd = [sys.executable, '-u', '-m', 'torch.distributed.launch', *args]
    print(cmd)
    try:
        process = subprocess.Popen(cmd, env=os.environ)
        process.wait()
    except Exception as e:
        logger.log('Error starting launcher', Text.danger)
        raise e

    if process.returncode != 0:
        logger.log('Launcher failed', Text.danger)
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


def _monitor():
    from labml.internal.computer import process

    process.run()


def main():
    parser = argparse.ArgumentParser(description='LabML CLI')
    parser.add_argument('command', choices=['dashboard', 'capture', 'launch', 'monitor'])
    parser.add_argument('args', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.command == 'dashboard':
        _open_dashboard()
    elif args.command == 'capture':
        _capture(args.args)
    elif args.command == 'launch':
        _launch(args.args)
    elif args.command == 'monitor':
        _monitor()
    else:
        raise ValueError('Unknown command', args.command)
