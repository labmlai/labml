import time
from typing import List, Dict, Set, Optional, Any

from labml_db import Model, Index, Key

from . import job
from . import run

JobResponse = Dict[str, str]

ONLINE_TIME_GAP = 60


class Computer(Model['Computer']):
    computer_uuid: str
    sessions: Set[str]
    active_runs: Set[str]
    deleted_runs: Set[str]
    pending_jobs: Dict[str, Key['job.Job']]
    completed_jobs: Dict[str, Key['job.Job']]
    last_online: float

    @classmethod
    def defaults(cls):
        return dict(computer_uuid='',
                    sessions=set(),
                    active_runs=set(),
                    deleted_runs=set(),
                    pending_jobs={},
                    completed_jobs={},
                    last_online=None
                    )

    @property
    def is_online(self) -> bool:
        return self.last_online and (time.time() - self.last_online) <= ONLINE_TIME_GAP

    def get_sessions(self) -> List[str]:
        return list(self.sessions)

    def get_deleted_runs(self) -> List[str]:
        return list(self.deleted_runs)

    def update_last_online(self) -> None:
        self.last_online = time.time()
        self.save()

    def is_method_repeated(self, method: str) -> bool:
        if method not in job.NON_REPEATED_METHODS:
            return False

        for job_uuid, job_key in self.pending_jobs.items():
            j = job_key.load()
            if j.is_non_repeated and j.method == method:
                return True

        return False

    def create_job(self, method: str, data: Dict[str, str]) -> 'job.Job':
        assert self.is_online, 'computer is not online'
        assert not self.is_method_repeated(method), 'non repeated method in the job queue'

        j = job.create(method, data)

        self.pending_jobs[j.job_uuid] = j.key
        self.save()

        return j

    def get_completed_job(self, job_uuid: str) -> Optional['job.Job']:
        if job_uuid in self.completed_jobs:
            job_key = self.completed_jobs[job_uuid]

            j = job_key.load()

            return j

        return None

    def get_pending_jobs(self) -> List['job.JobDict']:
        res = []
        for k, v in self.pending_jobs.items():
            j = v.load()
            res.append(j.to_data())

        return res

    def sync_runs(self, runs: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        active = []
        deleted = []
        unknown = []
        for run_data in runs:
            run_uuid = run_data['uuid']
            r = run.get(run_uuid)

            if r:
                r.sync_run(**run_data)
                active.append(run_uuid)
            elif run_uuid in self.deleted_runs:
                deleted.append(run_uuid)
            else:
                unknown.append(run_uuid)

        return {'active': active,
                'deleted': deleted,
                'unknown': unknown}

    def sync_jobs(self, responses: List[JobResponse]) -> None:
        for response in responses:
            job_uuid = response['uuid']
            status = response['status']
            data = response.get('data', {})

            if job_uuid in self.pending_jobs:
                j = self.pending_jobs[job_uuid].load()
                j.update_job(status, data)

                if j.is_completed:
                    self.pending_jobs.pop(job_uuid)
                    self.completed_jobs[job_uuid] = j.key

                    self.save()

            runs = data.get('runs', [])
            for run_data in runs:
                run_uuid = run_data.get('uuid')
                r = run.get(run_uuid)
                r.sync_run(**run_data)

    def get_data(self):
        return {
            'computer_uuid': self.computer_uuid,
            'sessions': self.get_sessions(),
        }


class ComputerIndex(Index['Computer']):
    pass


def get_or_create(computer_uuid: str) -> Computer:
    computer_key = ComputerIndex.get(computer_uuid)

    if not computer_key:
        computer = Computer(computer_uuid=computer_uuid,
                            )
        computer.save()
        ComputerIndex.set(computer_uuid, computer.key)

        return computer

    return computer_key.load()


def add_session(computer_uuid: str, session_uuid: str) -> None:
    if not computer_uuid:
        return

    c = get_or_create(computer_uuid)

    c.sessions.add(session_uuid)
    c.save()


def remove_session(computer_uuid: str, session_uuid: str) -> None:
    if not computer_uuid:
        return

    c = get_or_create(computer_uuid)

    if session_uuid in c.sessions:
        c.sessions.remove(session_uuid)
        c.save()


def add_run(computer_uuid: str, run_uuid: str) -> None:
    if not computer_uuid:
        return

    c = get_or_create(computer_uuid)

    c.active_runs.add(run_uuid)
    c.save()


def remove_run(computer_uuid: str, run_uuid: str) -> None:
    if not computer_uuid:
        return

    c = get_or_create(computer_uuid)

    if run_uuid in c.active_runs:
        c.active_runs.remove(run_uuid)
        c.deleted_runs.add(run_uuid)
        c.save()
