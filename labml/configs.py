from typing import List, Callable, overload, Union

from labml.internal.configs.base import Configs as _Configs
from labml.internal.configs.config_item import ConfigItem
from labml.utils.errors import ConfigsError


class BaseConfigs(_Configs):
    r"""
    You should sub-class this class to create your own configurations
    """
    pass


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
    if isinstance(name, ConfigItem):
        config_class = name.configs_class
    elif isinstance(name, list) and len(name) > 0 and isinstance(name[0], ConfigItem):
        config_class = name[0].config_class
    else:
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
    if isinstance(name, ConfigItem):
        config_class = name.configs_class
    elif isinstance(name, list) and len(name) > 0 and isinstance(name[0], ConfigItem):
        config_class = name[0].config_class
    else:
        raise ConfigsError('You need to pass config items to option')

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
