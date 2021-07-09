import subprocess
import threading
import time
from pathlib import Path

from typing.io import IO

from labml import monit
from labml_remote.execute import UIMode


class ExecutorThread(threading.Thread):
    process: subprocess.Popen

    def __init__(self, command: str, *, log_dir: Path, ui_mode: UIMode = UIMode.dots):
        super().__init__(daemon=False)
        self.ui_mode = ui_mode
        self.stdout_path = log_dir / 'stdout.log'
        self.stderr_path = log_dir / 'stderr.log'
        self.exit_code_path = log_dir / 'exit_code'
        self.command = command
        self.exit_code = 0

    def _read(self, stream: IO, path: Path, is_err: bool):
        if not stream.readable():
            return

        with open(str(path), 'ab') as f:
            while stream.readable():
                data = stream.read(1024)
                if len(data) == 0:
                    break
                f.write(data)
                self.ui_mode.on_bytes(data, is_err=is_err)

    def _read_stdout(self):
        self._read(self.process.stdout, self.stdout_path, False)

    def _read_stderr(self):
        self._read(self.process.stderr, self.stderr_path, True)

    def run(self):
        self.process = subprocess.Popen(self.command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        while True:
            self._read_stdout()
            self._read_stderr()
            if self.process.poll() is not None:
                break
            time.sleep(0.1)
        self.exit_code = self.process.returncode
        with open(str(self.exit_code_path), 'w') as f:
            f.write(str(self.exit_code))

        self._read_stdout()
        self._read_stderr()
        self.ui_mode.end()


class LocalExecutor:
    def __init__(self):
        pass

    def background(self, command: str, *, log_dir: Path, ui_mode: UIMode = UIMode.dots):
        thread = ExecutorThread(command,
                                log_dir=log_dir, ui_mode=ui_mode)
        thread.start()

        return 0

    def stream(self, command: str, *, log_dir: Path, ui_mode: UIMode = UIMode.dots, is_silent=True):
        with monit.section(f'Exec: {command}', is_silent=is_silent):
            thread = ExecutorThread(command,
                                    log_dir=log_dir, ui_mode=ui_mode)
            thread.start()
            thread.join()
            if thread.exit_code != 0:
                monit.fail()

            return thread.exit_code
