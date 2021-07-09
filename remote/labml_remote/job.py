import io
import sys
from pathlib import Path
from typing import Dict, Optional, List, Iterable

import yaml

from labml import logger
from labml.logger import Text
from labml_remote.configs import Configs
from labml_remote.execute import UIMode
from labml_remote.server import Server, SERVERS


class Job:
    def __init__(self, *, job_id: str, job_key: str, server: Server, command: str, env_vars: Dict[str, str],
                 tags: List[str],
                 pid: Optional[int] = None,
                 exit_code: Optional[int] = None,
                 started: bool = False,
                 stopped: bool = False, ):
        self.tags = set(tags)
        self.job_key = job_key
        self.env_vars = env_vars
        self.command = command
        self.server = server
        self.job_id = job_id
        self.path = Configs.get().project_jobs_folder / self.job_id
        self.pid = pid
        self.exit_code = exit_code
        self.started = started
        self.stopped = stopped
        self.out_tail_offset = -1
        self.err_tail_offset = -1
        if not self.path.exists():
            self.path.mkdir(parents=True)
            self.save()

        self.update_stopped()

    @property
    def running(self):
        return self.started and not self.stopped

    def update_stopped(self):
        if self.stopped:
            return

        file = self.path / 'stopped'
        if not file.exists():
            return

        self.stopped = True
        self.save()

    def to_dict(self):
        return {'job_id': self.job_id,
                'job_key': self.job_key,
                'server': self.server.conf.name,
                'command': self.command,
                'env_vars': self.env_vars,
                'tags': list(self.tags),
                'pid': self.pid,
                'exit_code': self.exit_code,
                'started': self.started,
                'stopped': self.stopped}

    @staticmethod
    def from_dict(data: Dict) -> 'Job':
        server = SERVERS[data.pop('server')]
        return Job(server=server, **data)

    def save(self):
        with open(str(self.path / 'job.yaml'), 'w') as f:
            f.write(yaml.dump(self.to_dict(), default_flow_style=False))

    def start(self):
        from labml_remote.util import get_env_vars
        res = self.server.script('job.sh', {
            'run_command': self.command,
            'job_id': self.job_id,
            'environment_variables': get_env_vars(self.env_vars)
        }, ui_mode=UIMode.none, is_background=False, is_eval=True)
        self.pid = int(res.out)
        self.exit_code = int(res.exit_code)
        self.started = True
        logger.log('Started job: ', (self.job_id, Text.value))
        logger.log('pid: ',
                   (str(self.pid), Text.meta), ' exit code: ',
                   (str(self.exit_code), Text.meta))
        self.save()

    def _tail_file(self, path: Path, tail_offset: int, file):
        with open(str(path), 'rb') as f:
            end = f.seek(0, io.SEEK_END)
            if tail_offset == -1:
                tail_offset = max(0, end - 400)
            f.seek(tail_offset)
            c = f.read()
            print(c.decode('utf-8'), end='', file=file)

        return end

    def tail(self):
        try:
            self.out_tail_offset = self._tail_file(self.path / 'job.out', self.out_tail_offset, sys.stdout)
        except FileNotFoundError:
            pass
        try:
            self.err_tail_offset = self._tail_file(self.path / 'job.err', self.err_tail_offset, sys.stderr)
        except FileNotFoundError:
            pass

    def has_tags(self, tags: List[str]):
        for t in tags:
            if t not in self.tags:
                return False

        return True


class JobCollection:
    _keys: Dict[str, Job]
    _jobs: Dict[str, Job]

    def __init__(self):
        self._jobs = {}
        self._keys = {}
        self.load_all()

    def __iter__(self):
        keys_list = list(self._jobs.keys())
        keys_list.sort(key=lambda k: int(self[k].job_key))
        return iter(keys_list)

    def __getitem__(self, job_id: str) -> Job:
        if job_id not in self._jobs:
            self.load(job_id)

        return self._jobs[job_id]

    def all(self) -> List[Job]:
        res = list(self._jobs.values())
        res.sort(key=lambda j: int(j.job_key))
        return res

    def filter_by_tags(self, tags: List[str], jobs: Optional[Iterable[Job]] = None) -> List[Job]:
        if jobs is None:
            jobs = self.all()

        return [j for j in jobs if j.has_tags(tags)]

    def filter_out_by_tags(self, tags: List[str], jobs: Optional[Iterable[Job]] = None) -> List[Job]:
        if jobs is None:
            jobs = self.all()

        return [j for j in jobs if not j.has_tags(tags)]

    def filter_running(self, jobs: Optional[Iterable[Job]] = None) -> List[Job]:
        if jobs is None:
            jobs = self.all()

        return [j for j in jobs if j.started and not j.stopped]

    def by_key(self, key: str):
        return self._keys[key]

    def job_keys(self):
        return list(self._keys.keys())

    def _next_key(self):
        keys = [int(k) for k in self._keys.keys()]
        if not keys:
            return '1'
        return str(max(keys) + 1)

    def create(self, server_id: str, command: str, env_vars: Dict[str, str], tags: List[str]):
        from uuid import uuid1
        job = Job(job_id=uuid1().hex, job_key=self._next_key(), server=SERVERS[server_id],
                  command=command, env_vars=env_vars, tags=tags)
        self._jobs[job.job_id] = job
        self._keys[job.job_key] = job
        return job

    def load(self, job_id: str):
        path = Configs.get().project_jobs_folder / job_id / 'job.yaml'
        with open(str(path), 'r') as f:
            self._jobs[job_id] = Job.from_dict(yaml.load(f.read(), Loader=yaml.FullLoader))
            self._keys[self._jobs[job_id].job_key] = self._jobs[job_id]

    def load_all(self):
        if not Configs.get().project_jobs_folder.exists():
            return
        for p in Configs.get().project_jobs_folder.iterdir():
            if p.name not in self._jobs:
                self.load(p.name)

    def start_status_watcher(self, check: bool = True):
        watching = set()
        if check:
            for _job in JOBS.all():
                if not _job.running:
                    continue
                if '__watch__' not in _job.tags:
                    continue
                watching.add(_job.server.conf.name)

        for k in SERVERS:
            if k in watching:
                continue
            s = SERVERS[k]

            s.copy_script(s.template_script('watch.py', {}), 'watch.py')
            self.create(k, f'python {s.remote_scripts_path}/watch.py', {},
                        ['__hidden__', '__watch__']).start()


JOBS = JobCollection()
