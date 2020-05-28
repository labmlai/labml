from typing import Optional

from labml.internal.configs.base import Configs as _Configs
from labml.internal.configs.config_item import ConfigItem


class BaseConfigs(_Configs):
    r"""
    You should sub-class this class to create your own configurations
    """
    pass


def option(config_item: any, option: Optional[str] = None):
    r"""
    An alternative to :meth:`BaseConfigs.calc`.
    """
    assert isinstance(config_item, ConfigItem)

    return config_item.calc(option)
