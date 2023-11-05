from typing import List

from labml.internal.computer.projects.sync import SyncRuns
from labml.internal.manage import runs as manage_runs

SYNC_RUNS = SyncRuns()


def delete_runs(*, runs: List[str]):
    paths = [r.path for r in SYNC_RUNS.get_runs(runs)]
    for p in paths:
        manage_runs.remove_run(p)

    return 'success', {}


def clear_checkpoints(*, runs: List[str]):
    runs = SYNC_RUNS.get_runs(runs)
    for r in runs:
        manage_runs.clear_checkpoints(r.path)
    for r in runs:
        r.scan()
    return 'success', {'runs': [r.to_dict() for r in runs]}


def call_sync():
    SYNC_RUNS.sync()
    return 'success', {}


METHODS = {
    'delete_runs': delete_runs,
    'clear_checkpoints': clear_checkpoints,
    'call_sync': call_sync,
}
