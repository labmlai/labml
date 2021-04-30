import json
from pathlib import Path

from labml.internal import util
from labml.internal.experiment.experiment_run import RunInfo

COMPLETED_STATUSES = {'completed', 'interrupted', 'crashed'}


class RunSummary:
    def __init__(self, path: Path):
        self.uuid = str(path.stem)
        self.path = path
        from labml.internal.computer.configs import computer_singleton
        self.cache_path = computer_singleton().runs_cache / self.uuid

        self.complete = False
        self.size = 0
        self.size_tensorboard = 0
        self.size_checkpoints = 0

        self.load_cache()
        if not self.complete:
            self.scan()

    def load_cache(self):
        if not self.cache_path.exists():
            return

        with open(str(self.cache_path), 'r') as f:
            data = util.yaml_load(f.read())

        if not data or data['path'] != str(self.path):
            return

        self.complete = data['complete']
        self.size = data['size']
        self.size_tensorboard = data['size_tensorboard']
        self.size_checkpoints = data['size_checkpoints']

    def to_dict(self):
        return {
            'uuid': self.uuid,
            'path': str(self.path),
            'complete': self.complete,
            'size': self.size,
            'size_tensorboard': self.size_tensorboard,
            'size_checkpoints': self.size_checkpoints,
        }

    def save_cache(self):
        with open(str(self.cache_path), 'w') as f:
            f.write(util.yaml_dump(self.to_dict()))

    def clear_cache(self):
        if self.cache_path.exists():
            self.cache_path.unlink()

    @staticmethod
    def _folder_size(path: Path):
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())

    def scan(self):
        run = RunInfo.from_path(self.path)

        self.size = self._folder_size(run.run_path)
        self.size_tensorboard = self._folder_size(run.tensorboard_log_path)
        self.size_checkpoints = self._folder_size(run.checkpoint_path)

        if run.run_log_path.exists():
            with open(str(run.run_log_path), 'r') as f:
                lines = f.read().split('\n')
                for l in lines:
                    if not l:
                        continue
                    d = json.loads(l)
                    if d['status'] in COMPLETED_STATUSES:
                        self.complete = True

        self.save_cache()
