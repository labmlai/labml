from typing import Optional, Set, Dict, overload

from labml.configs import BaseConfigs
from labml.internal.experiment import \
    create_experiment as _create_experiment, \
    experiment_singleton as _experiment_singleton, \
    has_experiment
from labml.internal.experiment.experiment_run import get_configs as _get_configs
from labml.internal.monitor import monitor_singleton as monitor

AVAILABLE_WRITERS = {'screen', 'app', 'file'}


def generate_uuid() -> str:
    from uuid import uuid1
    return uuid1().hex


def worker():
    """
    A worker like a data loader
    """
    if has_experiment():
        _experiment_singleton().worker()


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
           distributed_rank: int = 0,
           distributed_world_size: int = 0,
           distributed_main_rank: int = 0,
           disable_screen: bool = False):
    r"""
    Create an experiment

    Keyword Arguments:
        name (str, optional): name of the experiment
        python_file (str, optional): path of the Python file that
            created the experiment
        comment (str, optional): a short description of the experiment
        writers (Set[str], optional): list of writers to write stat to.
            Defaults to ``{'screen', 'app'}``.
        ignore_callers: (Set[str], optional): list of files to ignore when
            automatically determining ``python_file``
        tags (Set[str], optional): set of tags for experiment
        distributed_rank (int, optional): rank if this is a distributed training session
        distributed_world_size (int, optional): world_size if this is a distributed training session
    """

    if writers is None:
        writers = {'screen', 'app'}

    for w in writers:
        if w not in AVAILABLE_WRITERS:
            raise ValueError(f'Unknown writer: {w}')

    if disable_screen and 'screen' in writers:
        writers.remove('screen')

    if ignore_callers is None:
        ignore_callers = set()

    if uuid is None:
        import os
        if 'RUN_UUID' in os.environ:
            uuid = os.environ['RUN_UUID']

    if distributed_world_size > 0:
        if uuid is None:
            raise ValueError('You must provide a run UUID for distributed training sessions')

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
                       distributed_rank=distributed_rank,
                       distributed_world_size=distributed_world_size,
                       distributed_main_rank=distributed_main_rank,
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


def start(global_step: int = 0):
    r"""
    Starts the experiment.
    Run it using ``with`` statement and it will monitor and report, experiment completion
    and exceptions.
    """

    return _experiment_singleton().start()


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


def record(*,
           name: Optional[str] = None,
           comment: Optional[str] = None,
           writers: Set[str] = None,
           tags: Optional[Set[str]] = None,
           exp_conf: Dict[str, any] = None,
           lab_conf: Dict[str, any] = None,
           app_url: str = None,
           distributed_rank: int = 0,
           distributed_world_size: int = 0,
           disable_screen: bool = False):
    r"""
    This combines :func:`create`, :func:`configs` and :func:`start`.

    Keyword Arguments:
        name (str, optional): name of the experiment
        comment (str, optional): a short description of the experiment
        writers (Set[str], optional): list of writers to write stat to.
            Defaults to ``{'screen', 'app'}``.
        tags (Set[str], optional): Set of tags for experiment
        exp_conf (Dict[str, any], optional): a dictionary of experiment configurations
        lab_conf (Dict[str, any], optional): a dictionary of configurations for LabML.
         Use this if you want to change default configurations such as ``app_track_url``, and
         ``data_path``.
        app_url (str, optional): a shortcut to provide LabML app url
         instead of including it in ``lab_conf``. You can set this with :func:`labml.lab.configure`,
         `or with a configuration file for the entire project <../guide/installation_setup.html>`_.
        distributed_rank (int, optional): rank if this is a distributed training session
        distributed_world_size (int, optional): world_size if this is a distributed training session
    """

    if app_url is not None:
        if lab_conf is None:
            lab_conf = {}
        lab_conf['app_url'] = app_url

    if lab_conf is not None:
        from labml.internal.lab import lab_singleton as _internal
        _internal().set_configurations(lab_conf)

    create(name=name,
           python_file=None,
           comment=comment,
           writers=writers,
           ignore_callers=None,
           tags=tags,
           distributed_rank=distributed_rank,
           distributed_world_size=distributed_world_size,
           disable_screen=disable_screen)

    if exp_conf is not None:
        configs(exp_conf)

    return start()
