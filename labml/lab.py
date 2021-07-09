from pathlib import PurePath, Path
from typing import Dict

from labml.internal.lab import lab_singleton as _internal


def get_path() -> Path:
    r"""
    Get the path to the root of the project
    """
    return _internal().path


def get_data_path() -> Path:
    r"""
    Get the path to the data folder
    """
    return _internal().data_path


def get_experiments_path() -> Path:
    r"""
    Get the path to the root of experiment logs
    """
    return _internal().experiments


def configure(configurations: Dict[str, any]):
    r"""
    Set top level configurations through Python.
    You can set the configurations set through ``.labml.yaml``.

    `See the guide for details. <https://github.com/labmlai/labml/blob/master/guides/labml_yaml_file.md>`_
    """
    _internal().set_configurations(configurations)
