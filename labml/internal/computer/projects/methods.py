from typing import List


def start_tensorboard(*, runs: List[str]):
    return {'url': 'https://labml.ai'}


def delete_runs(*, runs: List[str]):
    return 'deleted'


def clear_checkpoints(*, runs: List[str]):
    return 'deleted'


def call_sync():
    return 'synced'


METHODS = {
    'start_tensorboard': start_tensorboard,
    'delete_runs': delete_runs,
    'clear_checkpoints': clear_checkpoints,
    'call_sync': call_sync,
}
