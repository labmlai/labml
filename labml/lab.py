from pathlib import PurePath
from typing import Dict

from labml.internal.lab import lab_singleton as _internal


def get_path() -> PurePath:
    return _internal().path


def get_data_path() -> PurePath:
    return _internal().data_path


def get_experiments_path() -> PurePath:
    return _internal().experiments


def configure(configurations: Dict[str, any]):
    _internal().set_configurations(configurations)
