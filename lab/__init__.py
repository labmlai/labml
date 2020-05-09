from pathlib import PurePath

from lab.internal.lab import lab_singleton as _internal


def get_data_path() -> PurePath:
    return _internal().data_path


def get_experiments_path() -> PurePath:
    return _internal().experiments
