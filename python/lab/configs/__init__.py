import functools

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


l = ListAppend(None, None)


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


class ConfigProcessor:
    def __init__(self, configs):
        assert isinstance(configs, Configs)

        classes = _get_classes(type(configs))

        self.values = {}
        self.properties = {}
        self.options = {}
        self.list_appends = {}

        for c in classes:
            for k, v in c.__dict__.items():
                self.collect_property(k, v)

        for k, v in configs.__dict__.items():
            self.collect_property(k, v)

        # collect from UI

    def collect_property(self, k, v):
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

    def topological_sort(self):
        pass

    def compute_values(self):
        pass
