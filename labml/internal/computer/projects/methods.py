from typing import List


def start_tensorboard(*, run_uuids: List[str]):
    return 'started'


def delete_runs(*, run_uuids: List[str]):
    return 'deleted'


def clear_checkpoints(*, run_uuids: List[str]):
    return 'deleted'


METHODS = {
    'start_tensorboard': start_tensorboard,
    'delete_runs': delete_runs,
    'clear_checkpoints': clear_checkpoints,
}
