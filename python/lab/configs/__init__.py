import functools
import inspect

from typing import List, NamedTuple

import typing


class Configs:
    pass


class Option(NamedTuple):
    func: typing.Callable
    config_name: str
    option_name: str


class ListAppend(NamedTuple):
    func: typing.Callable
    list_name: str


def option(config_name: str, option_name: str):
    def wrapper(func):
        return Option(func, config_name, option_name)

    return wrapper


def append(list_name: str):
    def wrapper(func):
        return ListAppend(func, list_name)

    return wrapper


def _get_classes(class_: type) -> List[type]:
    classes = [class_]
    level = [class_]
    next_level = []

    while len(level) > 0:
        for b in class_.__bases__:
            if b == object:
                continue
            next_level.append(b)
        classes += next_level
        level = next_level
        next_level = []

    classes.reverse()

    return classes


class ConfigPropertyException(ValueError):
    def __init__(self, func: typing.Callable):
        file = inspect.getfile(func)
        source = inspect.getsource(func)
        super().__init__(f"{file}\n{source}\n")


class MissingConfigException(ValueError):
    def __init__(self, func: typing.Callable, arg: str):
        file = inspect.getfile(func)
        source = inspect.getsource(func)
        super().__init__(f"Missing: {arg}\n{file}\n{source}\n")


class ConfigProcessor:
    def __init__(self, configs):
        assert isinstance(configs, Configs)

        classes = _get_classes(type(configs))

        self.configs = {}
        self.values = {}
        self.properties = {}
        self.options = {}
        self.list_appends = {}

        for c in classes:
            for k, v in c.__dict__.items():
                self.__collect_property(k, v)

        for k, v in configs.__dict__.items():
            self.__collect_property(k, v)

        # collect from UI

    def __collect_property(self, k, v):
        if type(v) == Option:
            if k not in self.options:
                self.options[k] = []
            self.options[k].append(v)
        elif type(v) == ListAppend:
            if k not in self.list_appends:
                self.list_appends[k] = []
            self.list_appends[k].append(v)
        elif callable(v):
            self.properties[k] = v
        else:
            self.values[k] = v

    def __check_properties(self):
        """
        Checks all the `properties`, `options` and `list_appends`
        """

        for p in self.properties.values():
            self.__check_callable(p)
        for opts in self.options.values():
            for p in opts:
                self.__check_callable(p)
        for appends in self.list_appends.values():
            for p in appends:
                self.__check_callable(p)

    def __check_callable(self, func: typing.Callable):
        arg_spec = inspect.getfullargspec(func)
        func_type = type(func)

        if func_type != type and arg_spec.args != []:
            raise ConfigPropertyException(func)

        if func_type == type and arg_spec.args != ['self']:
            raise ConfigPropertyException(func)

        if arg_spec.varargs is not None:
            raise ConfigPropertyException(func)

        if arg_spec.varkw is not None:
            raise ConfigPropertyException(func)

        if arg_spec.defaults is not None:
            raise ConfigPropertyException(func)

        if arg_spec.kwonlydefaults is not None:
            raise ConfigPropertyException(func)

        for arg in arg_spec.kwonlyargs:
            if (arg not in self.values) and (arg not in self.properties):
                raise MissingConfigException(func, arg)

    def __build_graph(self):
        visited = {}


    def topological_sort(self):
        pass

    def compute_values(self):
        pass
