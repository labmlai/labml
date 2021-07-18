import os
import time
from io import BufferedReader
from pathlib import Path
from queue import Queue, Empty
from subprocess import Popen, PIPE
from threading import Thread
from typing import List, Optional

from labml.logger import Text

from labml import logger

from labml.internal.experiment.experiment_run import RunInfo
from labml.internal.util import rm_tree


def enqueue_output(out: BufferedReader, queue: Queue):
    for line in iter(out.readline, b''):
        queue.put(line.decode('utf-8'))
    out.close()


def get_output(out: BufferedReader):
    q = Queue()
    thread = Thread(target=enqueue_output, args=(out, q))
    thread.daemon = True  # thread dies with the program
    thread.start()

    lines = []
    while True:
        try:
            line = q.get(timeout=30 if len(lines) == 0 else 2)
        except Empty:
            break
        else:
            lines.append(line)

    return lines


class TensorBoardStarter:
    pipe: Optional[Popen]

    def __init__(self, symlink_path: Path,
                 port: int = 6006, visible_port: int = 6006,
                 protocol: str = 'http', host: str = 'localhost'):
        self.visible_port = visible_port
        self.host = host
        self.protocol = protocol
        self.port = port
        self.symlink_path = symlink_path
        self.pipe = None

    @property
    def url(self):
        return f'{self.protocol}://{self.host}:{self.visible_port}/'

    def _create_symlink_folder(self):
        if self.symlink_path.exists():
            rm_tree(self.symlink_path)

        self.symlink_path.mkdir(parents=True)

    def start(self, runs: List[Path]):
        if self.pipe is not None:
            self.pipe.kill()

        self._create_symlink_folder()

        for p in runs:
            run = RunInfo.from_path(p)

            os.symlink(run.tensorboard_log_path, self.symlink_path / run.uuid)

        self.pipe = Popen(['tensorboard',
                           f'--logdir={self.symlink_path}',
                           '--port', f'{self.port}',
                           '--bind_all'],
                          env=os.environ.copy(),
                          stderr=PIPE)

        time.sleep(5)
        output = ''.join(get_output(self.pipe.stderr))
        if output.find('Press CTRL+C to quit') != -1:
            logger.log([('Tensorboard: ', Text.meta), (output, Text.subtle)])
            return True, output
        else:
            logger.log([('Failed to start Tensorboard: ', Text.warning), (output, Text.subtle)])
            return False, output

    def __del__(self):
        if self.pipe is not None:
            self.pipe.kill()


def _test():
    from labml.internal.computer.configs import computer_singleton
    from labml import lab
    from labml.internal.lab import lab_singleton
    import time

    lab_singleton().set_path(str(Path(os.path.abspath(__file__)).parent.parent.parent.parent))

    tb = TensorBoardStarter(computer_singleton().tensorboard_symlink_dir)

    # for k, v in os.environ.items():
    #     print(k, v)

    res = tb.start([
        lab.get_path() / 'logs' / 'sample' / '68233e98cb5311eb9aa38d17b08f3a1d',
    ])

    print(res)

    time.sleep(100)


if __name__ == '__main__':
    _test()
