from typing import List, Callable, overload, Union, Tuple

from labml.internal.configs.base import Configs as _Configs
from labml.internal.configs.config_item import ConfigItem
from labml.utils.errors import ConfigsError


class BaseConfigs(_Configs):
    r"""
    You should sub-class this class to create your own configurations
    """
    pass


def _get_config_class(name: any):
    if isinstance(name, ConfigItem):
        return name.configs_class
    elif isinstance(name, list) and len(name) > 0 and isinstance(name[0], ConfigItem):
        return name[0].configs_class
    else:
        return None


@overload
def option(name: Union[any, List[any]]):
    ...


@overload
def option(name: Union[any, List[any]], option_name: str):
    ...


@overload
def option(name: Union[any, List[any]], pass_params: List[any]):
    ...


@overload
def option(name: Union[any, List[any]], option_name: str, pass_params: List[any]):
    ...


def option(name: Union[any, List[any]], *args: any):
    r"""
    Use this as a decorator to register configuration options.

    This has multiple overloads

    .. function:: option(config_item: Union[any, List[any]])
        :noindex:

    .. function:: option(config_item: Union[any, List[any]], option_name: str)
        :noindex:

    .. function:: option(config_item: Union[any, List[any]], pass_params: List[any])
        :noindex:

    .. function:: option(config_item: Union[any, List[any]], option_name: str, pass_params: List[any])
        :noindex:

    Arguments:
        name: the configuration item or a list of items.
            If it is a list of items the function should return
            tuple.

        option_name (str, optional): name of the option.
            If not provided it will be derived from the
            function name.

        pass_params (list, optional): list of params to be passed.
            If not provided the configs object is passed.
            If provided the corresponding calculated configuration items
            will be passed to the function
    """
    config_class = _get_config_class(name)
    if config_class is None:
        raise ConfigsError('You need to pass config items to option')

    option_name = None
    pass_params = None
    for arg in args:
        if isinstance(arg, str):
            option_name = arg
        elif isinstance(arg, list):
            pass_params = arg

    return config_class.calc(name, option_name, pass_params)


@overload
def calculate(name: Union[any, List[any]], func: Callable):
    ...


@overload
def calculate(name: Union[any, List[any]], option_name: str, func: Callable):
    ...


@overload
def calculate(name: Union[any, List[any]],
              pass_params: List[any], func: Callable):
    ...


@overload
def calculate(name: Union[any, List[any]], option_name: str,
              pass_params: List[any], func: Callable):
    ...


def calculate(name: any, *args: any):
    r"""
    Use this to register configuration options.

    This has multiple overloads

    .. function:: calculate(name: Union[any, List[any]], func: Callable)
        :noindex:

    .. function:: calculate(name: Union[any, List[any]], option_name: str, func: Callable)
        :noindex:

    .. function:: calculate(name: Union[any, List[any]], pass_params: List[any], func: Callable)
        :noindex:

    .. function:: calculate(name: Union[any, List[any]], option_name: str, pass_params: List[any], func: Callable)
        :noindex:

    Arguments:
        name: the configuration item or a list of items.
            If it is a list of items the function should return
            tuple.

        func: the function to calculate the configuration

        option_name (str, optional): name of the option.
            If not provided it will be derived from the
            function name.

        pass_params (list, optional): list of params to be passed.
            If not provided the configs object is passed.
            If provided the corresponding calculated configuration items
            will be passed to the function
    """
    config_class = _get_config_class(name)
    if config_class is None:
        raise ConfigsError('You need to pass config items to calculate')

    option_name = None
    pass_params = None
    func = None
    for arg in args:
        if isinstance(arg, str):
            option_name = arg
        elif isinstance(arg, list):
            pass_params = arg
        elif type(arg) == type:
            func = arg
        else:
            func = arg

    if func is None:
        raise ConfigsError('You need to pass the function that calculates the configs')

    return config_class.calc_wrap(func, name, option_name, pass_params)


def hyperparams(*args: any, is_hyperparam=True):
    r"""
    Identifies configuration as (or not) hyper-parameters

    Arguments:
        *args: list of configurations
        is_hyperparam (bool, optional): whether the provided configuration
            items are hyper-parameters. Defaults to ``True``.
    """

    for arg in args:
        config_class = _get_config_class(arg)
        if config_class is None:
            raise ConfigsError('You need to pass config items to set hyperparams')
        config_class.set_hyperparams(arg, is_hyperparam=is_hyperparam)


def meta_config(*args: any, is_meta=True):
    r"""
    Identifies configuration as meta parameter

    Arguments:
        *args: list of configurations
        is_meta (bool, optional): whether the provided configuration
            items are meta. Defaults to ``True``.
    """

    for arg in args:
        config_class = _get_config_class(arg)
        if config_class is None:
            raise ConfigsError('You need to pass config items to set hyperparams')
        config_class.set_meta(arg, is_meta=is_meta)


def aggregate(name: any, option_name: str, *args: Tuple[any, any]):
    r"""
    Aggregate configs

    Arguments:
        name: name of the aggregate
        option_name: aggregate option name
        *args: list of configs to be aggregated
    """

    config_class = _get_config_class(name)
    if config_class is None:
        raise ConfigsError('You need to pass config item to aggregate')

    config_class.aggregate(name, option_name, *args)
