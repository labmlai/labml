import inspect
from typing import List, Callable, Union, Optional, TYPE_CHECKING

from labml.internal.configs.config_item import ConfigItem
from labml.internal.configs.utils import get_config_names

if TYPE_CHECKING:
    from labml.internal.configs.base import Configs


class SetupFunction:
    func: Callable
    config_name: str

    def __check_type(self):
        key, pos = 0, 0
        spec: inspect.Signature = inspect.signature(self.func)
        params: List[inspect.Parameter] = list(spec.parameters.values())

        for p in params:
            if p.kind == p.POSITIONAL_OR_KEYWORD:
                pos += 1
            elif p.kind == p.KEYWORD_ONLY:
                key += 1
            else:
                raise RuntimeError(f"Only positional or keyword only arguments should be accepted: "
                                   f"{self.config_name} - {self.func.__name__}")

        assert pos >= 1

    def __get_option_name(self, option_name: str):
        if option_name is not None:
            return option_name
        else:
            return self.func.__name__

    def __init__(self, func, *,
                 option_name: Optional[str] = None,
                 config_names: Union[str, ConfigItem, List[ConfigItem], List[str]]):
        self.secondary_attributes = {}
        self.func = func
        self.option_name = self.__get_option_name(option_name)
        self.config_names = get_config_names('setup', self.func.__name__, config_names)

        self.__check_type()

    def __call__(self, configs: Configs):
        self.func(configs)
