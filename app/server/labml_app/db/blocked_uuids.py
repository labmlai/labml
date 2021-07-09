from labml_db import Index

from . import run
from . import session


class BlockedRunIndex(Index['BlockedRun']):
    pass


class BlockedSessionIndex(Index['BlockedSession']):
    pass


def add_blocked_run(r: 'run.Run') -> None:
    BlockedRunIndex.set(r.run_uuid, r.key)


def add_blocked_session(s: 'session.Session') -> None:
    BlockedSessionIndex.set(s.session_uuid, s.key)


def is_run_blocked(run_uuid: str) -> bool:
    run_key = BlockedRunIndex.get(run_uuid)

    return run_key is not None


def is_session_blocked(session_uuid: str) -> bool:
    session_key = BlockedSessionIndex.get(session_uuid)

    return session_key is not None


def remove_blocked_run(run_uuid: str) -> None:
    BlockedRunIndex.delete(run_uuid)


def remove_blocked_session(session_uuid: str) -> None:
    BlockedSessionIndex.delete(session_uuid)
