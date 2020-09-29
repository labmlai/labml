import inspect
import warnings
from typing import List, Callable, Set, Union, Optional

from labml.internal.configs.config_item import ConfigItem

from labml.internal.configs.dependency_parser import DependencyParser


class SetupFunction:
    func: Callable
    dependencies: Set[str]
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

    def __get_dependencies(self):
        parser = DependencyParser(self.func)
        if parser.is_referenced:
            raise RuntimeError(f"{self.func.__name__} should only use attributes of configs")
        required = set(parser.required)

        for c in self.config_names:
            if c in required:
                required.remove(c)

        return required

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
        self.check_string_names = True
        self.config_names = self.__get_config_names(config_names)

        self.__check_type()
        self.dependencies = self.__get_dependencies()

    def __call__(self, configs: 'Configs'):
        self.func(configs)
