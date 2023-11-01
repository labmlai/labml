import time
from typing import List, Dict, Set

from labml_db import Model, Index

JobResponse = Dict[str, str]

ONLINE_TIME_GAP = 60


class Computer(Model['Computer']):
    computer_uuid: str
    sessions: Set[str]
    active_runs: Set[str]
    deleted_runs: Set[str]
    last_online: float

    @classmethod
    def defaults(cls):
        return dict(computer_uuid='',
                    sessions=set(),
                    active_runs=set(),
                    deleted_runs=set(),
                    last_online=None
                    )

    @property
    def is_online(self) -> bool:
        return self.last_online and (time.time() - self.last_online) <= ONLINE_TIME_GAP

    def get_sessions(self) -> List[str]:
        return list(self.sessions)

    def get_deleted_runs(self) -> List[str]:
        return list(self.deleted_runs)

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
