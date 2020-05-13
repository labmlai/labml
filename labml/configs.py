from typing import Callable

from labml.internal.configs.base import Configs as _Configs
from labml.internal.configs.config_item import ConfigItem


class BaseConfigs(_Configs):
    r"""
    You should sub-class this class to create your own configurations
    """
    pass


def option(config_item: any):
    assert isinstance(config_item, ConfigItem)

    def wrapper(func: Callable):
        return config_item(func)

    return wrapper
