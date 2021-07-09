from typing import List

from labml.internal.computer.configs import computer_singleton
from labml.internal.computer.projects.sync import SyncRuns
from labml.internal.manage import runs as manage_runs
from labml.internal.manage.tensorboard import TensorBoardStarter

SYNC_RUNS = SyncRuns()
TENSORBOARD_STARTER = TensorBoardStarter(
    computer_singleton().tensorboard_symlink_dir,
    computer_singleton().tensorboard_port,
    computer_singleton().tensorboard_visible_port,
    computer_singleton().tensorboard_protocol,
    computer_singleton().tensorboard_host,
)


def start_tensorboard(*, runs: List[str]):
    paths = [r.path for r in SYNC_RUNS.get_runs(runs)]
    ret, msg = TENSORBOARD_STARTER.start(paths)
    if ret:
        return 'success', {
            'url': TENSORBOARD_STARTER.url,
            'message': msg,
        }
    else:
        return 'fail', {
            'message': msg,
        }


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
    'start_tensorboard': start_tensorboard,
    'delete_runs': delete_runs,
    'clear_checkpoints': clear_checkpoints,
    'call_sync': call_sync,
}
