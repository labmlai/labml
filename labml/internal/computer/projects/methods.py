from typing import List

from labml.internal.computer.projects.sync import SyncRuns

SYNC_RUNS = SyncRuns()


def start_tensorboard(*, runs: List[str]):
    return {'url': 'https://labml.ai'}


def delete_runs(*, runs: List[str]):
    return 'deleted'


def clear_checkpoints(*, runs: List[str]):
    return 'deleted'


def call_sync():
    SYNC_RUNS.sync()
    return {'synced': True}


METHODS = {
    'start_tensorboard': start_tensorboard,
    'delete_runs': delete_runs,
    'clear_checkpoints': clear_checkpoints,
    'call_sync': call_sync,
}
