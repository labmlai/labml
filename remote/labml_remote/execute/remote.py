import threading
import time
from pathlib import Path
from typing import NamedTuple, Optional

from paramiko import SSHClient

from labml import monit
from labml_remote.execute import UIMode


class ExecutorThread(threading.Thread):
    def __init__(self, client: SSHClient, command: str, *,
                 log_dir: Path,
                 ui_mode: UIMode = UIMode.dots):
        super().__init__(daemon=False)
        self.ui_mode = ui_mode
        self.stdout_path = log_dir / 'stdout.log'
        self.stderr_path = log_dir / 'stderr.log'
        self.exit_code_path = log_dir / 'exit_code'
        self.command = command
        self.client = client
        self.exit_code = 0
        self.channel = None

    def _read_stdout(self):
        if not self.channel.recv_ready():
            return

        with open(str(self.stdout_path), 'ab') as f:
            while self.channel.recv_ready():
                data = self.channel.recv(1024)
                f.write(data)
                self.ui_mode.on_bytes(data, is_err=False)

    def _read_stderr(self):
        if not self.channel.recv_stderr_ready():
            return

        with open(str(self.stderr_path), 'ab') as f:
            while self.channel.recv_stderr_ready():
                data = self.channel.recv_stderr(1024)
                f.write(data)
                self.ui_mode.on_bytes(data, is_err=True)

    def run(self):
        self.channel = self.client.get_transport().open_session()
        self.channel.exec_command(self.command)

        while True:
            self._read_stdout()
            self._read_stderr()
            if self.channel.exit_status_ready():
                break
            time.sleep(0.1)
        self.exit_code = self.channel.recv_exit_status()
        with open(str(self.exit_code_path), 'w') as f:
            f.write(str(self.exit_code))

        self._read_stdout()
        self._read_stderr()
        self.ui_mode.end()


class EvalResult(NamedTuple):
    exit_code: int
    out: str
    err: str


class RemoteExecutor:
    client: SSHClient

    def __init__(self, client: SSHClient):
        self.client = client

    def eval(self, command: str, *, log_dir: Optional[Path], is_silent=True) -> EvalResult:
        with monit.section(f'Exec: {command}', is_silent=is_silent):
            stdin, stdout, stderr = self.client.exec_command(command)
            out = stdout.read()
            err = stderr.read()
            exit_code = stdout.channel.recv_exit_status()
            if log_dir:
                with open(str(log_dir / 'stdout.log'), 'wb') as f:
                    f.write(out)
                with open(str(log_dir / 'stderr.log'), 'wb') as f:
                    f.write(err)
                with open(str(log_dir / 'exit_code'), 'w') as f:
                    f.write(str(exit_code))

            if exit_code != 0:
                monit.fail()

            return EvalResult(exit_code, out.decode('utf-8').strip(), err.decode('utf-8').strip())

    def background(self, command: str, *, log_dir: Path, ui_mode: UIMode = UIMode.dots):
        thread = ExecutorThread(self.client, command,
                                log_dir=log_dir, ui_mode=ui_mode)
        thread.start()

        return EvalResult(0, '', '')

    def stream(self, command: str, *, log_dir: Path, ui_mode: UIMode = UIMode.dots, is_silent=True):
        with monit.section(f'Exec: {command}', is_silent=is_silent):
            thread = ExecutorThread(self.client, command,
                                    log_dir=log_dir, ui_mode=ui_mode)
            thread.start()
            thread.join()
            if thread.exit_code != 0:
                monit.fail()

            return EvalResult(thread.exit_code, '', '')
