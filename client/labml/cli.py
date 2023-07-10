import argparse
import os
import subprocess
import sys
import threading
import time
from typing import List

from labml import logger, experiment
from labml.experiment import generate_uuid
from labml.internal.api import ApiCaller, SimpleApiDataSource
from labml.internal.api.logs import ApiLogs
from labml.internal.api.url import ApiUrlHandler
from labml.internal.lab import get_api_url
from labml.logger import Text
from labml.utils.validators import ip_validator
from typing.io import IO

COMMAND_APP_SERVER = 'app-server'
COMMAND_CAPTURE = 'capture'
COMMAND_LAUNCH = 'launch'
COMMAND_MONITOR = 'monitor'
COMMAND_SERVICE = 'service'
COMMAND_SERVICE_RUN = 'service-run'
COMMAND_DASHBOARD = 'dashboard'


def _open_dashboard():
    try:
        import labml_dashboard
    except (ImportError, ModuleNotFoundError):
        logger.log("Cannot import ", ('labml_dashboard', Text.highlight), '.')
        logger.log('Install with ',
                   ('pip install labml_dashboard', Text.value))
        return

    labml_dashboard.start_server()


def _start_app_server(ip: str, port: int):
    try:
        import labml_app
    except (ImportError, ModuleNotFoundError):
        logger.log("Cannot import ", ('labml_app', Text.highlight), '.')
        logger.log('Install with ',
                   ('pip install labml-app', Text.value))
        return

    labml_app.start_server(ip=ip, port=port)


class ExecutorThread(threading.Thread):
    process: subprocess.Popen

    def __init__(self, command: str, api_logs: ApiLogs):
        super().__init__(daemon=False)
        self.api_logs = api_logs
        self.command = command
        self.exit_code = 0

    def _read(self, stream: IO, name: str):
        buffer = ''
        while stream.readable():
            data = stream.read(1)
            if len(data) == 0:
                break
            buffer += data
            print(data, end='')
            if '\n' in buffer or len(buffer) > 100:
                self.api_logs.outputs(**{name: buffer})
                buffer = ''
        if len(buffer) > 0:
            self.api_logs.outputs(**{name: buffer})

    def _read_stdout(self):
        self._read(self.process.stdout, 'stdout_')

    def _read_stderr(self):
        self._read(self.process.stderr, 'stderr_')

    def run(self):
        shell = os.environ.get('SHELL', '/bin/sh')

        self.process = subprocess.Popen(
            self.command,
            # [shell, '-i', '-c', self.command],
            encoding='utf-8',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            env=os.environ.copy(),
            executable=shell,
            bufsize=0,
        )

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
    base_url = get_api_url('track')
    if base_url is None:
        raise RuntimeError(f'Please specify labml_server environment variable. '
                           f'How to setup a labml server https://github.com/labmlai/labml/tree/master/app')

    api_caller = ApiCaller(base_url, {'run_uuid': generate_uuid()},
                           timeout_seconds=120)
    api_logs = ApiLogs()
    data = {
        'name': 'Capture',
        'comment': ' '.join(args),
        'time': time.time()
    }

    api_caller.add_handler(ApiUrlHandler(True, 'Monitor output at '))
    api_caller.has_data(SimpleApiDataSource(data))
    api_logs.set_api(api_caller, frequency=0)

    logger.log('Start capturing...', Text.meta)
    if args:
        thread = ExecutorThread(' '.join(args), api_logs)
        thread.start()
        thread.join()
    else:
        buffer = ''
        stdin = sys.stdin
        while stdin.readable():
            data = stdin.read(1)
            if len(data) == 0:
                break
            print(data, end='')
            buffer += data
            if '\n' in buffer or len(buffer) > 100:
                api_logs.outputs(stdout_=buffer)
                buffer = ''
        if len(buffer) > 0:
            api_logs.outputs(stdout_=buffer)

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

    cmd = [sys.executable, '-u', '-m', 'torch.distributed.run', *args]
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
    from labml.internal.computer.configs import computer_singleton

    process.run(True, computer_singleton().web_api.open_browser)


def _service():
    from labml.internal.computer.service import service_singleton

    service_singleton().set_token()
    service_singleton().create()


def _service_run():
    from labml.internal.computer import process

    process.run(True, False)


def main():
    parser = argparse.ArgumentParser(description='labml.ai CLI', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser = parser.add_subparsers(title='command', dest='command', required=True)
    app_server_parser = subparser.add_parser(COMMAND_APP_SERVER, help='Start a local instance of labml server',
                                             formatter_class=parser.formatter_class)
    app_server_parser.add_argument('--ip', type=ip_validator, default='0.0.0.0', help='IP to bind the server')
    app_server_parser.add_argument('--port', type=int, default=5005, help='Port to run the server on')

    capture_parser = subparser.add_parser(COMMAND_CAPTURE, help='Create an experiment manually')
    capture_parser.add_argument('args', nargs=argparse.REMAINDER)

    launch_parser = subparser.add_parser(COMMAND_LAUNCH, help='Start a distributed training session')
    launch_parser.add_argument('args', nargs=argparse.REMAINDER)

    subparser.add_parser(COMMAND_MONITOR, help='Start hardware monitoring')
    subparser.add_parser(COMMAND_SERVICE, help='Setup and start a service for hardware monitoring')
    subparser.add_parser(COMMAND_SERVICE_RUN, help='Start hardware monitoring (for internal use)')
    subparser.add_parser(COMMAND_DASHBOARD, help='Open the labml dashboard (deprecated)')

    args = parser.parse_args()

    if args.command == COMMAND_DASHBOARD:
        logger.log([('labml dashboard', Text.heading),
                    (' is deprecated. Please use labml.ai app instead\n', Text.danger),
                    ('https://github.com/labmlai/labml/tree/master/app', Text.value)])
        _open_dashboard()
    elif args.command == COMMAND_APP_SERVER:
        _start_app_server(ip=args.ip, port=args.port)
    elif args.command == COMMAND_CAPTURE:
        _capture(args.args)
    elif args.command == COMMAND_LAUNCH:
        _launch(args.args)
    elif args.command == COMMAND_MONITOR:
        _monitor()
    elif args.command == COMMAND_SERVICE:
        _service()
    elif args.command == COMMAND_SERVICE_RUN:
        _service_run()
    else:
        raise ValueError('Unknown command', args.command)


def _test_capture():
    _capture([])


if __name__ == '__main__':
    main()
