from pathlib import PurePath
from typing import Dict, Optional


def new_run(python_file: PurePath, *,
            configs: Optional[Dict[str, any]] = None,
            comment: Optional[str] = None):
    r"""
    This starts a new experiment by running ``main`` function defined in ``python_file``.

    Arguments:
        python_file (PurePath): path of the Python script

    Keyword Arguments:
        configs (Dict[str, any], optional): a dictionary of configurations
        comment (str, optional): comment to identify the experiment
    """
    from labml.internal.manage.process import new_run as _new_run
    _new_run(python_file, configs=configs, comment=comment)


def new_run_process(python_file: PurePath, *,
                    configs: Optional[Dict[str, any]] = None,
                    comment: Optional[str] = None):
    r"""
    This starts a new experiment in a separate process by running ``main`` function defined in ``python_file``.

    Arguments:
        python_file (PurePath): path of the Python script

    Keyword Arguments:
        configs (Dict[str, any], optional): a dictionary of configurations
        comment (str, optional): comment to identify the experiment
    """
    from labml.internal.manage.process import new_run_process as _new_run_process
    _new_run_process(python_file, configs=configs, comment=comment)
