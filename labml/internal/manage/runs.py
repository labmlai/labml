from pathlib import Path
from typing import Optional, Generator

from labml import lab
from labml.logger import inspect


def get_runs() -> Generator[Path, None, None]:
    for exp_path in lab.get_experiments_path().iterdir():
        if exp_path.name.startswith('_'):
            continue
        for run_path in exp_path.iterdir():
            yield run_path


def get_experiments() -> Generator[Path, None, None]:
    for exp_path in lab.get_experiments_path().iterdir():
        if exp_path.name.startswith('_'):
            continue

        yield exp_path


def get_run_by_uuid(run_uuid: str) -> Optional[Path]:
    for exp_path in get_experiments():
        run_path = exp_path / run_uuid
        if run_path.exists():
            return run_path

    return None


def get_runs_by_experiment_path(exp_path: Path) -> Generator[Path, None, None]:
    for run_path in exp_path.iterdir():
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


# ===TESTS===
def _test_checkpoints_by_uuid(run_uuid: str):
    run_path = get_run_by_uuid(run_uuid)
    inspect(list(get_checkpoints(run_path)))


def _test_remove_run_by_uuid(run_uuid: str):
    run_path = get_run_by_uuid(run_uuid)
    remove_run(run_path)


if __name__ == '__main__':
    inspect(list(get_runs()))
    # _test_checkpoints_by_uuid('794ca6f21c1f11ebba97acde48001122')
    # _test_remove_run_by_uuid('1443e69c1c2111eb8e8bacde48001122')
