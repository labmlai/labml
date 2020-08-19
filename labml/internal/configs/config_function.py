import inspect
import warnings
from enum import Enum
from typing import List, Callable, cast, Set, Union, Optional
from typing import TYPE_CHECKING

from .config_item import ConfigItem
from .dependency_parser import DependencyParser
from ...utils.errors import ConfigsError

if TYPE_CHECKING:
    from .base import Configs


class FunctionKind(Enum):
    pass_configs = 'pass_configs'
    pass_kwargs = 'pass_kwargs'
    pass_nothing = 'pass_nothing'
    pass_params = 'pass_params'


class ConfigFunction:
    func: Callable
    kind: FunctionKind
    dependencies: Set[str]
    config_names: Union[str, List[str]]
    option_name: str
    params: List[inspect.Parameter]
    pass_params: Optional[List[ConfigItem]]

    def __get_type(self):
        key, pos = 0, 0

        for p in self.params:
            if p.kind == p.POSITIONAL_OR_KEYWORD:
                pos += 1
            elif p.kind == p.KEYWORD_ONLY:
                key += 1
            else:
                raise ConfigsError(f"Only positional or keyword only arguments should be accepted: "
                                   f"{self.config_names} - {self.option_name}")

        if self.pass_params is not None:
            if key > 0:
                raise ConfigsError(f"No keyword arguments are supported when passing configs: "
                                   f"{self.config_names} - {self.option_name}")

            if pos != len(self.pass_params):
                raise ConfigsError('Number of arguments to function should match the number of '
                                   'configs to be passed: '
                                   f"{self.config_names} - {self.option_name}")
            return FunctionKind.pass_params

        if pos == 1:
            if key != 0:
                raise ConfigsError(
                    'No keyword arguments are allowed when passing a config object: '
                    f"{self.config_names} - {self.option_name}")
            return FunctionKind.pass_configs
        elif pos == 0 and key == 0:
            return FunctionKind.pass_nothing
        else:
            warnings.warn("Use configs object, because it's easier to refactor, find usage etc",
                          FutureWarning, stacklevel=4)
            if pos != 0:
                raise ConfigsError(
                    f"{self.config_names} - {self.option_name}")
            return FunctionKind.pass_kwargs

    def __get_dependencies(self):
        if self.kind == FunctionKind.pass_configs:
            parser = DependencyParser(self.func)
            assert not parser.is_referenced, \
                f"{self.func.__name__} should only use attributes of configs"
            self.secondary_attributes = parser.secondary_attributes
            return parser.required
        elif self.kind == FunctionKind.pass_kwargs:
            return {p.name for p in self.params}
        elif self.kind == FunctionKind.pass_params:
            return {p.key for p in self.pass_params}
        elif self.kind == FunctionKind.pass_nothing:
            return set()
        else:
            assert False

    def __get_option_name(self, option_name: str):
        if option_name is not None:
            return option_name
        else:
            return self.func.__name__

    def __get_config_names(self, config_names: Union[str, ConfigItem, List[ConfigItem], List[str]]):
        if config_names is None:
            warnings.warn("Use @Config.[name]", FutureWarning, 4)
            return self.func.__name__
        elif type(config_names) == str:
            if self.check_string_names:
                warnings.warn("Use @Config.[name] instead of '[name]'", FutureWarning, 4)
            return config_names
        elif type(config_names) == ConfigItem:
            return config_names.key
        else:
            assert type(config_names) == list
            assert len(config_names) > 0
            if type(config_names[0]) == str:
                warnings.warn("Use @Config.[name] instead of '[name]'", FutureWarning, 4)
                return config_names
            else:
                assert type(config_names[0]) == ConfigItem
                return [c.key for c in config_names]

    def __get_params(self):
        func_type = type(self.func)

        if func_type == type:
            init_func = cast(object, self.func).__init__
            spec: inspect.Signature = inspect.signature(init_func)
            params: List[inspect.Parameter] = list(spec.parameters.values())
            assert len(params) > 0
            assert params[0].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD, self.config_names
            assert params[0].name == 'self'
            return params[1:]
        else:
            spec: inspect.Signature = inspect.signature(self.func)
            params: List[inspect.Parameter] = list(spec.parameters.values())
            return params

    def __init__(self, func, *,
                 config_names: Union[str, ConfigItem, List[ConfigItem], List[str]],
                 option_name: str,
                 pass_params: Optional[List[ConfigItem]] = None,
                 check_string_names: bool = True):
        self.secondary_attributes = {}
        self.func = func
        self.check_string_names = check_string_names
        self.pass_params = pass_params
        self.config_names = self.__get_config_names(config_names)
        self.option_name = self.__get_option_name(option_name)

        self.params = self.__get_params()

        self.kind = self.__get_type()
        self.dependencies = self.__get_dependencies()

    def __call__(self, configs: 'Configs'):
        if self.kind == FunctionKind.pass_configs:
            if len(self.params) == 1:
                return self.func(configs)
            else:
                return self.func()
        elif self.kind == FunctionKind.pass_kwargs:
            kwargs = {p.name: configs.__getattribute__(p.name) for p in self.params}
            return self.func(**kwargs)
        elif self.kind == FunctionKind.pass_params:
            args = [configs.__getattribute__(p.key) for p in self.pass_params]
            return self.func(*args)
        elif self.kind == FunctionKind.pass_nothing:
            return self.func()
        else:
            assert False
