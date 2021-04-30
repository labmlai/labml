from pathlib import Path
from typing import Optional, Generator

from labml.logger import inspect


def _is_valid_path(path: Path):
    if path.name.startswith('_') or path.name.startswith('.'):
        return False
    else:
        return True


def get_runs(experiments_path: Path) -> Generator[Path, None, None]:
    for exp_path in get_experiments(experiments_path):
        for run_path in exp_path.iterdir():
            if _is_valid_path(run_path):
                yield run_path


def get_experiments(experiments_path: Path) -> Generator[Path, None, None]:
    if not experiments_path.exists():
        return

    for exp_path in experiments_path.iterdir():
        if not _is_valid_path(exp_path):
            continue

        yield exp_path


def get_run_by_uuid(experiments_path: Path, run_uuid: str) -> Optional[Path]:
    for exp_path in get_experiments(experiments_path):
        run_path = exp_path / run_uuid
        if run_path.exists():
            return run_path

    return None


def get_runs_by_experiment_path(exp_path: Path) -> Generator[Path, None, None]:
    for run_path in exp_path.iterdir():
        if _is_valid_path(run_path):
            yield run_path


def get_checkpoints(run_path: Path) -> Generator[Path, None, None]:
    checkpoints_path = run_path / "checkpoints"
    if not checkpoints_path.exists():
        return

    for chk_path in checkpoints_path.iterdir():
        yield chk_path


def remove_run(run_path: Path):
    from labml.internal.util import rm_tree
    rm_tree(run_path)


def clear_checkpoints(run_path: Path):
    checkpoints = list(get_checkpoints(run_path))
    if not checkpoints:
        return

    checkpoint_steps = sorted([int(c.name) for c in checkpoints])

    if len(checkpoint_steps) <= 2:
        return

    checkpoint_steps = checkpoint_steps[:-2]
    checkpoints_path = run_path / "checkpoints"
    from labml.internal.util import rm_tree

    for c in checkpoint_steps:
        path = checkpoints_path / str(c)
        rm_tree(path)


# ===TESTS===
def _test_checkpoints_by_uuid(experiments_path: Path, run_uuid: str):
    run_path = get_run_by_uuid(experiments_path, run_uuid)
    inspect(list(get_checkpoints(run_path)))


def _test_remove_run_by_uuid(experiments_path: Path, run_uuid: str):
    run_path = get_run_by_uuid(experiments_path, run_uuid)
    remove_run(run_path)


def _test():
    from labml import lab
    inspect(list(get_runs(lab.get_experiments_path())))
    # _test_checkpoints_by_uuid(lab.get_experiments_path(), '794ca6f21c1f11ebba97acde48001122')
    # _test_remove_run_by_uuid(lab.get_experiments_path(), '1443e69c1c2111eb8e8bacde48001122')


if __name__ == '__main__':
    _test()
