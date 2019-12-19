from collections import OrderedDict
from pathlib import PurePath
from typing import List, Dict, Callable, Type, Set, Optional, \
    OrderedDict as OrderedDictType, Union, Any, Tuple

from lab import util
from .calculator import Calculator

_CALCULATORS = '_calculators'


class Configs:
    _calculators: Dict[str, List[Calculator]] = {}

    @classmethod
    def calc(cls, name: Union[str, List[str]] = None,
             option: str = None, *,
             is_append: bool = False):
        if _CALCULATORS not in cls.__dict__:
            cls._calculators = {}

        def wrapper(func: Callable):

            calc = Calculator(func, config_names=name, option_name=option, is_append=is_append)
            if type(calc.config_names) == str:
                config_names = [calc.config_names]
            else:
                config_names = calc.config_names

            for n in config_names:
                if n not in cls._calculators:
                    cls._calculators[n] = []
                cls._calculators[n].append(calc)

            return func

        return wrapper

    @classmethod
    def list(cls, name: str = None):
        return cls.calc(name, f"_{util.random_string()}", is_append=True)


def _get_base_classes(class_: Type[Configs]) -> List[Type[Configs]]:
    classes = [class_]
    level = [class_]
    next_level = []

    while len(level) > 0:
        for c in level:
            for b in c.__bases__:
                if b == object:
                    continue
                next_level.append(b)
        classes += next_level
        level = next_level
        next_level = []

    classes.reverse()

    return classes


RESERVED = {'calc', 'list'}


class ConfigProcessor:
    topological_order: List[str]
    stack: List[str]
    visited: Set[str]

    options: Dict[str, OrderedDictType[str, Calculator]]
    types: Dict[str, Type]
    values: Dict[str, any]
    list_appends: Dict[str, List[Calculator]]
    dependencies: Dict[str, Set[str]]

    is_computed: Set[str]
    is_top_sorted: Set[str]

    def __init__(self, configs, values: Dict[str, any] = None):
        assert isinstance(configs, Configs)

        classes = _get_base_classes(type(configs))

        self.values = {}
        self.types = {}
        self.options = {}
        self.list_appends = {}
        self.configs = configs

        for c in classes:
            for k, v in c.__annotations__.items():
                self.__collect_annotation(k, v)

            for k, v in c.__dict__.items():
                self.__collect_value(k, v)

        for c in classes:
            if _CALCULATORS in c.__dict__:
                for k, calcs in c.__dict__[_CALCULATORS].items():
                    assert k in self.types, k
                    for v in calcs:
                        self.__collect_calculator(k, v)

        for k, v in configs.__dict__.items():
            assert k in self.types
            self.__collect_value(k, v)

        if values is not None:
            for k, v in values.items():
                assert k in self.types
                self.__collect_value(k, v)

    @staticmethod
    def is_valid(key):
        if key.startswith('_'):
            return False

        if key in RESERVED:
            return False

        return True

    def __collect_value(self, k, v):
        if not self.is_valid(k):
            return

        self.values[k] = v
        if k not in self.types:
            self.types[k] = type(v)

    def __collect_annotation(self, k, v):
        if not self.is_valid(k):
            return

        self.types[k] = v

    def __collect_calculator(self, k, v: Calculator):
        if v.is_append:
            if k not in self.list_appends:
                self.list_appends[k] = []
            self.list_appends[k].append(v)
        else:
            if k not in self.options:
                self.options[k] = OrderedDict()
            self.options[k][v.option_name] = v

    def __get_property(self, key) -> Tuple[Any, Union[None, Calculator, List[Calculator]]]:
        if key in self.options:
            value = self.values[key]
            if value not in self.options[key]:
                return value, None
            return None, self.options[key][value]

        if key in self.list_appends:
            return None, [f for f in self.list_appends[key]]

        return self.values[key], None

    def __calculate_missing_values(self):
        for k in self.types:
            if k in self.values and self.values[k] is not None:
                continue

            if k in self.list_appends:
                continue

            if k in self.options:
                self.values[k] = next(iter(self.options[k].keys()))
                continue

            if type(self.types[k]) == type:
                self.options[k] = OrderedDict()
                self.options[k][k] = Calculator(self.types[k],
                                                config_names=k,
                                                option_name=k,
                                                is_append=False)
                self.values[k] = k
                continue

            assert k in self.values, f"Cannot compute {k}"

    def __get_dependencies(self, key) -> Set[str]:
        assert not (key in self.options and key in self.list_appends), \
            f"{key} in options and appends"

        if key in self.options:
            value = self.values[key]
            if value not in self.options[key]:
                return set()
            return self.options[key][value].dependencies

        if key in self.list_appends:
            dep = set()
            for func in self.list_appends[key]:
                dep = dep.union(func.dependencies)

            return dep

        return set()

    def __create_graph(self):
        self.dependencies = {}
        for k in self.types:
            self.dependencies[k] = self.__get_dependencies(k)

    def __add_to_topological_order(self, key):
        assert self.stack.pop() == key
        self.is_top_sorted.add(key)
        self.topological_order.append(key)

    def __traverse(self, key):
        for d in self.dependencies[key]:
            if d not in self.is_top_sorted:
                self.__add_to_stack(d)
                return

        self.__add_to_topological_order(key)

    def __add_to_stack(self, key):
        if key in self.is_top_sorted:
            return

        assert key not in self.visited, f"Cyclic dependency: {key}"

        self.visited.add(key)
        self.stack.append(key)

    def __dfs(self):
        while len(self.stack) > 0:
            key = self.stack[-1]
            self.__traverse(key)

    def __topological_sort(self, keys: List[str]):
        for k in keys:
            assert k not in self.is_top_sorted

        for k in keys:
            self.__add_to_stack(k)
            self.__dfs()

    def __set_configs(self, key, value):
        assert key not in self.is_computed
        self.is_computed.add(key)
        self.configs.__setattr__(key, value)

    def __compute(self, key):
        if key in self.is_computed:
            return

        value, funcs = self.__get_property(key)
        if value is not None:
            self.__set_configs(key, value)
        elif type(funcs) == list:
            self.__set_configs(key, [f(self.configs) for f in funcs])
        else:
            value = funcs(self.configs)
            if type(funcs.config_names) == str:
                self.__set_configs(key, value)
            else:
                for i, k in enumerate(funcs.config_names):
                    self.__set_configs(k, value[i])

    def __compute_values(self):
        for k in self.topological_order:
            if k not in self.is_computed:
                self.__compute(k)

    def calculate(self, run_order: Optional[List[Union[List[str], str]]]):
        if run_order is None:
            run_order = [list(self.types.keys())]

        for i in range(len(run_order)):
            keys = run_order[i]
            if type(keys) == str:
                run_order[i] = [keys]

        self.__calculate_missing_values()
        self.__create_graph()

        self.visited = set()
        self.stack = []
        self.is_top_sorted = set()
        self.topological_order = []
        self.is_computed = set()

        for keys in run_order:
            self.__topological_sort(keys)
            self.__compute_values()

    def save(self, configs_path: PurePath):
        configs = {
            'values': self.values,
            'options': {}
        }

        for k, opts in self.options.items():
            configs['options'][k] = [o for o in opts]

        with open(str(configs_path), "w") as file:
            file.write(util.yaml_dump(configs))
