from pathlib import Path
from typing import Optional, Set, Dict, List, Union, TYPE_CHECKING, overload

import numpy as np

from labml.configs import BaseConfigs
from labml.internal.experiment import \
    create_experiment as _create_experiment, \
    experiment_singleton as _experiment_singleton, \
    ModelSaver
from labml.internal.experiment.experiment_run import \
    get_configs as _get_configs

if TYPE_CHECKING:
    import torch


def save_checkpoint():
    r"""
    Saves model checkpoints
    """
    _experiment_singleton().save_checkpoint()


def get_uuid():
    r"""
    Returns the UUID of the current experiment run
    """

    return _experiment_singleton().run.uuid


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
        writers = {'sqlite', 'tensorboard', 'web_api'}

    if ignore_callers is None:
        ignore_callers = {}

    _create_experiment(name=name,
                       python_file=python_file,
                       comment=comment,
                       writers=writers,
                       ignore_callers=ignore_callers,
                       tags=tags)


def add_model_savers(savers: Dict[str, ModelSaver]):
    _experiment_singleton().checkpoint_saver.add_savers(savers)


def add_pytorch_models(models: Dict[str, 'torch.nn.Module']):
    """
    Set variables for saving and loading

    Arguments:
        models (Dict[str, torch.nn.Module]): a dictionary of torch modules
            used in the experiment.
            These will be saved with :func:`labml.experiment.save_checkpoint`
            and loaded with :func:`labml.experiment.load`.
    """

    from labml.internal.experiment.pytorch import add_models as _add_pytorch_models
    _add_pytorch_models(models)


def add_sklearn_models(models: Dict[str, any]):
    """
    .. warning::
        This is still experimental.

    Set variables for saving and loading

    Arguments:
        models (Dict[str, any]): a dictionary of SKLearn models
            These will be saved with :func:`labml.experiment.save_checkpoint`
            and loaded with :func:`labml.experiment.load`.
    """
    from labml.internal.experiment.sklearn import add_models as _add_sklearn_models
    _add_sklearn_models(models)


@overload
def configs(configs_dict: Dict[str, any]):
    ...


@overload
def configs(configs_dict: Dict[str, any], configs_override: Dict[str, any]):
    ...


@overload
def configs(configs: BaseConfigs):
    ...


@overload
def configs(configs: BaseConfigs, run_order: List[Union[List[str], str]]):
    ...


@overload
def configs(configs: BaseConfigs, *run_order: str):
    ...


@overload
def configs(configs: BaseConfigs, configs_override: Dict[str, any]):
    ...


@overload
def configs(configs: BaseConfigs, configs_override: Dict[str, any],
                      run_order: List[Union[List[str], str]]):
    ...


@overload
def configs(configs: BaseConfigs, configs_override: Dict[str, any],
                      *run_order: str):
    ...


def configs(*args):
    r"""
    Calculate configurations

    This has multiple overloads

    .. function:: configs(configs_dict: Dict[str, any])
        :noindex:

    .. function:: configs(configs_dict: Dict[str, any], configs_override: Dict[str, any])
        :noindex:

    .. function:: configs(configs: BaseConfigs)
        :noindex:

    .. function:: configs(configs: BaseConfigs, run_order: List[Union[List[str], str]])
        :noindex:

    .. function:: configs(configs: BaseConfigs, *run_order: str)
        :noindex:

    .. function:: configs(configs: BaseConfigs, configs_override: Dict[str, any])
        :noindex:

    .. function:: configs(configs: BaseConfigs, configs_override: Dict[str, any], run_order: List[Union[List[str], str]])
        :noindex:

    .. function:: configs(configs: BaseConfigs, configs_override: Dict[str, any], *run_order: str)
        :noindex:

    Arguments:
        configs (BaseConfigs, optional): configurations object
        configs_dict (Dict[str, any], optional): a dictionary of configs
        configs_override (Dict[str, any], optional): a dictionary of
            configs to be overridden
        run_order (List[Union[str, List[str]]], optional): list of
            configs to be calculated and the order in which they should be
            calculated. If not provided all configs will be calculated.
    """
    configs_override: Optional[Dict[str, any]] = None
    run_order: Optional[List[Union[List[str], str]]] = None
    idx = 1

    if isinstance(args[0], BaseConfigs):
        if idx < len(args) and isinstance(args[idx], dict):
            configs_override = args[idx]
            idx += 1

        if idx < len(args) and isinstance(args[idx], list):
            run_order = args[idx]
            if len(args) != idx + 1:
                raise RuntimeError("Invalid call to calculate configs")
            _experiment_singleton().calc_configs(args[0], configs_override, run_order)
        else:
            if idx == len(args):
                _experiment_singleton().calc_configs(args[0], configs_override, run_order)
            else:
                run_order = list(args[idx:])
                for key in run_order:
                    if not isinstance(key, str):
                        raise RuntimeError("Invalid call to calculate configs")
                _experiment_singleton().calc_configs(args[0], configs_override, run_order)
    elif isinstance(args[0], dict):
        if idx < len(args) and isinstance(args[idx], dict):
            configs_override = args[idx]
            idx += 1

        if idx != len(args):
            raise RuntimeError("Invalid call to calculate configs")

        _experiment_singleton().calc_configs_dict(args[0], configs_override)
    else:
        raise RuntimeError("Invalid call to calculate configs")


def start():
    r"""
    Starts the experiment.
    """
    _experiment_singleton().start()


def load_configs(run_uuid: str, *, is_only_hyperparam: bool = True):
    r"""
    Load configs of a previous run

    Arguments:
        run_uuid (str): if provided the experiment will start from
            a saved state in the run with UUID ``run_uuid``
    Keyword Arguments:
        is_only_hyperparam (bool, optional): if True all only the hyper parameters
            are returned
    """

    configs = _get_configs(run_uuid)
    values = {}
    for k, c in configs.items():
        is_hyperparam = c.get('is_hyperparam', None)
        is_explicit = c.get('is_explicitly_specified', False)

        if not is_only_hyperparam:
            values[k] = c['value']
        elif is_hyperparam is None and is_explicit:
            values[k] = c['value']
        elif is_hyperparam:
            values[k] = c['value']

    return values


def load(run_uuid: str,
         checkpoint: Optional[int] = None):
    r"""
    Loads and starts the run from a previous checkpoint.

    Arguments:
        run_uuid (str): experiment will start from
            a saved state in the run with UUID ``run_uuid``
        checkpoint (str, optional): if provided the experiment will start from
            given checkpoint. Otherwise it will start from the last checkpoint.
    """

    _experiment_singleton().start(run_uuid=run_uuid, checkpoint=checkpoint)


def load_models(models: List[str], run_uuid: str,
         checkpoint: Optional[int] = None):
    r"""
    Loads and starts the run from a previous checkpoint.

    Arguments:
        run_uuid (str): experiment will start from
            a saved state in the run with UUID ``run_uuid``
        checkpoint (str, optional): if provided the experiment will start from
            given checkpoint. Otherwise it will start from the last checkpoint.
    """

    _experiment_singleton().load_models(models=models, run_uuid=run_uuid, checkpoint=checkpoint)

def save_numpy(name: str, array: np.ndarray):
    r"""
    Saves a single numpy array. This is used to save processed data.
    """

    numpy_path = Path(_experiment_singleton().run.numpy_path)

    if not numpy_path.exists():
        numpy_path.mkdir(parents=True)
    file_name = name + ".npy"
    np.save(str(numpy_path / file_name), array)
