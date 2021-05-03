import os
from pathlib import Path
from subprocess import Popen, PIPE
from typing import List, Optional

from labml.logger import Text

from labml import logger

from labml.internal.experiment.experiment_run import RunInfo
from labml.internal.util import rm_tree


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

        output = self.pipe.stderr.readline().decode('utf-8')
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
    import time

    tb = TensorBoardStarter(computer_singleton().tensorboard_symlink_dir)

    # for k, v in os.environ.items():
    #     print(k, v)

    res = tb.start([
        lab.get_path() / 'logs' / 'sample' / '9f7970d6a98611ebbc6bacde48001122',
    ])

    print(res)

    time.sleep(100)


if __name__ == '__main__':
    _test()
