from typing import List, Dict, Type, Set, Optional, \
    OrderedDict as OrderedDictType, Union, Any, Tuple
from typing import TYPE_CHECKING

from .config_function import ConfigFunction

if TYPE_CHECKING:
    from . import Configs


class Calculator:
    options: Dict[str, OrderedDictType[str, ConfigFunction]]
    types: Dict[str, Type]
    values: Dict[str, any]
    list_appends: Dict[str, List[ConfigFunction]]

    configs: 'Configs'

    dependencies: Dict[str, Set[str]]
    topological_order: List[str]
    stack: List[str]
    visited: Set[str]
    is_computed: Set[str]
    is_top_sorted: Set[str]

    def __init__(self, *,
                 configs: 'Configs',
                 options: Dict[str, OrderedDictType[str, ConfigFunction]],
                 types: Dict[str, Type],
                 values: Dict[str, any],
                 list_appends: Dict[str, List[ConfigFunction]]):
        self.configs = configs
        self.options = options
        self.types = types
        self.values = values
        self.list_appends = list_appends

        self.visited = set()
        self.stack = []
        self.is_top_sorted = set()
        self.topological_order = []
        self.is_computed = set()

    def __get_property(self, key) -> Tuple[Any, Union[None, ConfigFunction, List[ConfigFunction]]]:
        if key in self.options:
            value = self.values[key]
            if value not in self.options[key]:
                return value, None
            return None, self.options[key][value]

        if key in self.list_appends:
            return None, [f for f in self.list_appends[key]]

        return self.values[key], None

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

    def __call__(self, run_order: Optional[List[Union[List[str], str]]]):
        if run_order is None:
            run_order = [list(self.types.keys())]

        for i in range(len(run_order)):
            keys = run_order[i]
            if type(keys) == str:
                run_order[i] = [keys]

        self.__create_graph()

        self.visited = set()
        self.stack = []
        self.is_top_sorted = set()
        self.topological_order = []
        self.is_computed = set()

        for keys in run_order:
            self.__topological_sort(keys)
            self.__compute_values()
