from pathlib import Path
from typing import Optional, Set, Dict, List, Union, TYPE_CHECKING, overload

import numpy as np

from labml.configs import BaseConfigs
from labml.internal.experiment import \
    create_experiment as _create_experiment, \
    experiment_singleton as _experiment_singleton, \
    ModelSaver
from labml.internal.monitor import monitor_singleton as monitor
from labml.internal.experiment.experiment_run import \
    get_configs as _get_configs

if TYPE_CHECKING:
    import torch


def generate_uuid() -> str:
    from uuid import uuid1
    return uuid1().hex


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
           uuid: Optional[str] = None,
           name: Optional[str] = None,
           python_file: Optional[str] = None,
           comment: Optional[str] = None,
           writers: Set[str] = None,
           ignore_callers: Set[str] = None,
           tags: Optional[Set[str]] = None,
           disable_screen: bool = False):
    r"""
    Create an experiment

    Keyword Arguments:
        name (str, optional): name of the experiment
        python_file (str, optional): path of the Python file that
            created the experiment
        comment (str, optional): a short description of the experiment
        writers (Set[str], optional): list of writers to write stat to.
            Defaults to ``{'tensorboard', 'sqlite', 'web_api'}``.
        ignore_callers: (Set[str], optional): list of files to ignore when
            automatically determining ``python_file``
        tags (Set[str], optional): Set of tags for experiment
    """

    if writers is None:
        writers = {'screen', 'sqlite', 'web_api'}
        from labml.internal.util.tensorboard_writer import has_tensorboard
        if has_tensorboard():
            writers.add('tensorboard')

    if disable_screen and 'screen' in writers:
        writers.remove('screen')

    if ignore_callers is None:
        ignore_callers = set()

    if uuid is None:
        uuid = generate_uuid()

    monitor().clear()
    _create_experiment(uuid=uuid,
                       name=name,
                       python_file=python_file,
                       comment=comment,
                       writers=writers,
                       ignore_callers=ignore_callers,
                       tags=tags,
                       is_evaluate=False)


def evaluate():
    r"""
    This should be used for evaluation of a saved experiment.
    This will not record anything.
    """

    monitor().clear()
    _create_experiment(uuid=generate_uuid(),
                       name=None,
                       python_file=None,
                       comment=None,
                       writers=set(),
                       ignore_callers=set(),
                       tags=None,
                       is_evaluate=True)


def distributed(rank: int, world_size: int):
    _experiment_singleton().distributed(rank, world_size)


def add_model_savers(savers: Dict[str, ModelSaver]):
    _experiment_singleton().checkpoint_saver.add_savers(savers)


@overload
def add_pytorch_models(**kwargs: 'torch.nn.Module'):
    ...


@overload
def add_pytorch_models(models: Dict[str, 'torch.nn.Module']):
    ...


def add_pytorch_models(*args, **kwargs):
    """
    Set variables for saving and loading

    Arguments:
        models (Dict[str, torch.nn.Module]): a dictionary of torch modules
            used in the experiment.
            These will be saved with :func:`labml.experiment.save_checkpoint`
            and loaded with :func:`labml.experiment.load`.
    """

    from labml.internal.experiment.pytorch import add_models as _add_pytorch_models

    if len(args) > 0:
        assert len(args) == 1
        assert isinstance(args[0], dict)
        _add_pytorch_models(args[0])
    else:
        _add_pytorch_models(kwargs)


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
def configs(conf_dict: Dict[str, any]):
    ...


@overload
def configs(conf_dict: Dict[str, any], conf_override: Dict[str, any]):
    ...


@overload
def configs(conf: BaseConfigs):
    ...


@overload
def configs(conf: BaseConfigs, conf_override: Dict[str, any]):
    ...


def configs(*args):
    r"""
    Calculate configurations

    This has multiple overloads

    .. function:: configs(conf_dict: Dict[str, any])
        :noindex:

    .. function:: configs(conf_dict: Dict[str, any], conf_override: Dict[str, any])
        :noindex:

    .. function:: configs(conf: BaseConfigs)
        :noindex:

    .. function:: configs(conf: BaseConfigs, run_order: List[Union[List[str], str]])
        :noindex:

    .. function:: configs(conf: BaseConfigs, *run_order: str)
        :noindex:

    .. function:: configs(conf: BaseConfigs, conf_override: Dict[str, any])
        :noindex:

    .. function:: configs(conf: BaseConfigs, conf_override: Dict[str, any], run_order: List[Union[List[str], str]])
        :noindex:

    .. function:: configs(conf: BaseConfigs, conf_override: Dict[str, any], *run_order: str)
        :noindex:

    Arguments:
        conf (BaseConfigs, optional): configurations object
        conf_dict (Dict[str, any], optional): a dictionary of configs
        conf_override (Dict[str, any], optional): a dictionary of
            configs to be overridden
    """
    conf_override: Optional[Dict[str, any]] = None
    conf = args[0]
    idx = 1

    if idx < len(args) and isinstance(args[idx], dict):
        conf_override = args[idx]
        idx += 1

    if len(args) != idx:
        raise RuntimeError("Invalid call to calculate configs")

    if isinstance(conf, BaseConfigs):
        _experiment_singleton().calc_configs(conf, conf_override)
    elif isinstance(conf, dict):
        _experiment_singleton().calc_configs(conf, conf_override)
    else:
        raise RuntimeError("Invalid call to calculate configs")


_load_run_uuid: Optional[str] = None
_load_checkpoint: Optional[int] = None


def start():
    r"""
    Starts the experiment.
    Run it using ``with`` statement and it will monitor and report, experiment completion
    and exceptions.
    """
    global _load_run_uuid
    global _load_checkpoint

    return _experiment_singleton().start(run_uuid=_load_run_uuid, checkpoint=_load_checkpoint)


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

    conf = _get_configs(run_uuid)
    if not conf:
        return {}
    values = {}
    for k, c in conf.items():
        is_hyperparam = c.get('is_hyperparam', None)
        is_explicit = c.get('is_explicitly_specified', False)

        if not is_only_hyperparam:
            values[k] = c['value']
        elif is_hyperparam is None and is_explicit:
            values[k] = c['value']
        elif is_hyperparam:
            values[k] = c['value']

    return values


def load(run_uuid: str, checkpoint: Optional[int] = None):
    r"""
    Loads a the run from a previous checkpoint.
    You need to separately call ``experiment.start`` to start the experiment.

    Arguments:
        run_uuid (str): experiment will start from
            a saved state in the run with UUID ``run_uuid``
        checkpoint (str, optional): if provided the experiment will start from
            given checkpoint. Otherwise it will start from the last checkpoint.
    """
    global _load_run_uuid
    global _load_checkpoint

    _load_run_uuid = run_uuid
    _load_checkpoint = checkpoint


def load_models(models: List[str], run_uuid: str, checkpoint: Optional[int] = None):
    r"""
    Loads and starts the run from a previous checkpoint.

    Arguments:
        models (List[str]): List of names of models to be loaded
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


def record(*,
           name: Optional[str] = None,
           comment: Optional[str] = None,
           writers: Set[str] = None,
           tags: Optional[Set[str]] = None,
           exp_conf: Dict[str, any] = None,
           lab_conf: Dict[str, any] = None,
           token: str = None,
           disable_screen: bool = False):
    r"""
    This is combines :func:`create`, :func:`configs` and :func:`start`.

    Keyword Arguments:
        name (str, optional): name of the experiment
        comment (str, optional): a short description of the experiment
        writers (Set[str], optional): list of writers to write stat to.
            Defaults to ``{'tensorboard', 'sqlite', 'web_api'}``.
        tags (Set[str], optional): Set of tags for experiment
        exp_conf (Dict[str, any], optional): a dictionary of experiment configurations
        lab_conf (Dict[str, any], optional): a dictionary of configurations for LabML.
         Use this if you want to change default configurations such as ``web_api``, and
         ``data_path``.
        token (str, optional): a shortcut to provide LabML mobile app token (or url - ``web_api``)
         instead of including it in ``lab_conf``. You can set this with :func:`labml.lab.configure`,
         `or with a configuration file for the entire project <../guide/installation_setup.html>`_.
    """

    if token is not None:
        if lab_conf is None:
            lab_conf = {}
        lab_conf['web_api'] = token

    if lab_conf is not None:
        from labml.internal.lab import lab_singleton as _internal
        _internal().set_configurations(lab_conf)

    create(name=name,
           python_file=None,
           comment=comment,
           writers=writers,
           ignore_callers=None,
           tags=tags,
           disable_screen=disable_screen)

    if exp_conf is not None:
        configs(exp_conf)

    return start()
