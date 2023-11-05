import json
import time
from pathlib import Path
from typing import List

from labml.internal import util
from labml.internal.experiment.experiment_run import RunInfo

COMPLETED_STATUSES = {'completed', 'interrupted', 'crashed'}

MINUTE = 60


class RunSummary:
    def __init__(self, uuid: str, paths: List[Path]):
        self.uuid = uuid
        self.paths = paths
        from labml.internal.computer.configs import computer_singleton
        self.cache_path = computer_singleton().runs_cache / self.uuid

        self.complete = False
        self.size = 0
        self.size_checkpoints = 0
        self.last_scanned = 0

        self.load_cache()
        if not self.complete and self.last_scanned < time.time() - MINUTE:
            self.scan()

    def load_cache(self):
        if not self.cache_path.exists():
            return

        with open(str(self.cache_path), 'r') as f:
            data = util.yaml_load(f.read())

        if not data or data['uuid'] != self.uuid:
            return

        self.complete = data['complete']
        self.size = data['size']
        self.size_checkpoints = data['size_checkpoints']
        self.last_scanned = data['last_scanned']

    def to_dict(self):
        return {
            'uuid': self.uuid,
            'paths': [str(p) for p in self.paths],
            'complete': self.complete,
            'size': self.size,
            'size_checkpoints': self.size_checkpoints,
            'last_scanned': self.last_scanned,
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
        self.size = 0
        self.size_checkpoints = 0
        self.complete = False

        for p in self.paths:
            run = RunInfo.from_path(p)

            self.size += self._folder_size(run.run_path)
            self.size_checkpoints += self._folder_size(run.checkpoint_path)

            if run.run_log_path.exists():
                with open(str(run.run_log_path), 'r') as f:
                    lines = f.read().split('\n')
                    for l in lines:
                        if not l:
                            continue
                        d = json.loads(l)
                        if d['status'] in COMPLETED_STATUSES:
                            self.complete = True

        self.last_scanned = time.time()

        self.save_cache()
