from pathlib import Path
from typing import Optional, Set, Dict, List, Union, TYPE_CHECKING

import numpy as np

from lab.configs import BaseConfigs
from lab.internal.experiment import \
    create_experiment as _create_experiment, \
    experiment_singleton as _experiment_singleton

if TYPE_CHECKING:
    import torch


def save_checkpoint():
    _experiment_singleton().save_checkpoint()


def create(*,
           name: Optional[str] = None,
           python_file: Optional[str] = None,
           comment: Optional[str] = None,
           writers: Set[str] = None,
           ignore_callers: Set[str] = None,
           tags: Optional[Set[str]] = None):
    r"""
    Create an experiment

    Keyword Arguments:
        name (str, optional): name of the experiment
        python_file (str, optional): path of the Python file that
            created the experiment
        comment (str, optional): a short description of the experiment
        writers (Set[str], optional): list of writers to write stat to.
            Defaults to ``{'tensorboard', 'sqlite'}``.
        ignore_callers: (Set[str], optional): list of files to ignore when
            automatically determining ``python_file``
        tags (Set[str], optional): Set of tags for experiment
    """

    if writers is None:
        writers = {'sqlite', 'tensorboard'}

    if ignore_callers is None:
        ignore_callers = {}

    _create_experiment(name=name,
                       python_file=python_file,
                       comment=comment,
                       writers=writers,
                       ignore_callers=ignore_callers,
                       tags=tags)


def add_pytorch_models(models: Dict[str, 'torch.nn.Module']):
    """
    Set variables for saving and loading

    Arguments:
        models (Dict[str, torch.nn.Module]): a dictionary of torch modules
            used in the experiment.
            These will be saved with :func:`lab.experiment.save_checkpoint`
            and loaded with :func:`lab.experiment.load`.
    """

    from lab.internal.experiment.pytorch import add_models as _add_pytorch_models
    _add_pytorch_models(models)


def add_sklearn_models(models: Dict[str, any]):
    """
    .. warning::
        This is still experimental.

    Set variables for saving and loading

    Arguments:
        models (Dict[str, any]): a dictionary of SKLearn models
            These will be saved with :func:`lab.experiment.save_checkpoint`
            and loaded with :func:`lab.experiment.load`.
    """
    from lab.internal.experiment.sklearn import add_models as _add_sklearn_models
    _add_sklearn_models(models)


def calculate_configs(
        configs: Optional[BaseConfigs],
        configs_dict: Optional[Dict[str, any]] = None,
        run_order: Optional[List[Union[List[str], str]]] = None):
    r"""
    Calculate configurations

    Arguments:
        configs (Configs, optional): configurations object
        configs_dict (Dict[str, any], optional): a dictionary of
            configs to be overridden
        run_order (List[Union[str, List[str]]], optional): list of
            configs to be calculated and the order in which they should be
            calculated. If ``None`` all configs will be calculated.
    """
    if configs_dict is None:
        configs_dict = {}

    _experiment_singleton().calc_configs(configs, configs_dict, run_order)


def start():
    r"""
    Starts the experiment.
    """
    _experiment_singleton().start()


def load(run_uuid: str,
         checkpoint: Optional[int] = None):
    r"""
    Loads and starts the experiment from a previous checkpoint.

    Keyword Arguments:
        run_uuid (str): if provided the experiment will start from
            a saved state in the run with UUID ``run_uuid``
        checkpoint (str, optional): if provided the experiment will start from
            given checkpoint. Otherwise it will start from the last checkpoint.
    """
    _experiment_singleton().start(run_uuid=run_uuid, checkpoint=checkpoint)


def save_numpy(name: str, array: np.ndarray):
    r"""
    Saves a single numpy array. This is used to save processed data.
    """
    numpy_path = Path(_experiment_singleton().run.numpy_path)

    if not numpy_path.exists():
        numpy_path.mkdir(parents=True)
    file_name = name + ".npy"
    np.save(str(numpy_path / file_name), array)
