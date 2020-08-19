from typing import List, Dict, Type, Set, Optional, Union, Any, Tuple
from typing import TYPE_CHECKING

from .config_function import ConfigFunction
from .eval_function import EvalFunction
from ... import logger
from ... import monit

if TYPE_CHECKING:
    from .base import Configs
    from .processor import ConfigProcessor


class Calculator:
    secondary_values: Dict[str, Dict[str, any]]
    evals: Dict[str, Dict[str, EvalFunction]]
    options: Dict[str, Dict[str, ConfigFunction]]
    types: Dict[str, Type]
    values: Dict[str, any]

    configs: 'Configs'

    dependencies: Dict[str, Set[str]]
    topological_order: List[str]
    stack: List[str]
    visited: Set[str]
    is_computed: Set[str]
    is_top_sorted: Set[str]
    config_processors: Dict[str, 'ConfigProcessor']
    secondary_attributes: Dict[str, Set[str]]

    def __init__(self, *,
                 configs: 'Configs',
                 options: Dict[str, Dict[str, ConfigFunction]],
                 evals: Dict[str, Dict[str, EvalFunction]],
                 types: Dict[str, Type],
                 values: Dict[str, any],
                 secondary_values: Dict[str, Dict[str, any]],
                 aggregate_parent: Dict[str, str]):
        self.secondary_values = secondary_values
        self.aggregate_parent = aggregate_parent
        self.evals = evals
        self.configs = configs
        self.options = options
        self.types = types
        self.values = values

        self.visited = set()
        self.stack = []
        self.is_top_sorted = set()
        self.topological_order = []
        self.is_computed = set()
        self.config_processors = {}
        self.secondary_attributes = {}

    def __get_property(self, key) -> Tuple[Any, Union[None, ConfigFunction, List[ConfigFunction]]]:
        if key in self.options:
            value = self.values[key]
            if value not in self.options[key]:
                return value, None
            return None, self.options[key][value]

        return self.values[key], None

    def __get_dependencies(self, key) -> Set[str]:
        if key in self.options:
            value = self.values[key]
            if value not in self.options[key]:
                return set()
            sa = self.options[key][value].secondary_attributes
            for dep, att in sa.items():
                if dep not in self.secondary_attributes:
                    self.secondary_attributes[dep] = set()
                self.secondary_attributes[dep] = self.secondary_attributes[dep].union(att)
            return self.options[key][value].dependencies

        if key in self.evals:
            value = self.values.get(key, None)
            if not value:
                value = 'default'
            if value not in self.evals[key]:
                return set()
            return self.evals[key][value].dependencies

        if key not in self.values:
            raise RuntimeError(f"Cannot compute {key}")

        return set()

    def __create_graph(self):
        self.dependencies = {}
        for k in self.types:
            self.dependencies[k] = self.__get_dependencies(k)
        for k in self.evals:
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

    def __set_configs(self, key, value, is_direct: bool):
        assert key not in self.is_computed
        from .base import Configs
        from .processor import ConfigProcessor
        if isinstance(value, Configs):
            primary = value.__dict__.get('_primary', None)
            processor = ConfigProcessor(value,
                                        self.secondary_values.get(key, None),
                                        is_directly_specified=is_direct)
            secondary_attributes = self.secondary_attributes.get(key, set())
            if primary:
                secondary_attributes.add(primary)
            processor(list(secondary_attributes))
            if primary:
                value = value.__dict__.get(primary)
            self.config_processors[key] = processor

        self.is_computed.add(key)
        self.configs.__setattr__(key, value)

    def __compute(self, key):
        if key in self.is_computed:
            return

        if key in self.evals:
            return

        value, funcs = self.__get_property(key)
        if funcs is None:
            self.__set_configs(key, value, is_direct=True)
        elif type(funcs) == list:
            self.__set_configs(key, [f(self.configs) for f in funcs], is_direct=False)
        else:
            s = monit.section(f'Prepare {key}', is_new_line=False)
            with s:
                value = funcs(self.configs)
            if s.get_estimated_time() >= 0.01:
                logger.log()
            else:
                logger.log(' ' * 100, is_new_line=False)

            if type(funcs.config_names) == str:
                self.__set_configs(key, value, is_direct=False)
            else:
                if not isinstance(value, tuple) or not isinstance(value, list):
                    raise RuntimeError(f"Expect a tuple of a list as results for {funcs.config_names}")
                if not len(value) != len(funcs.config_names):
                    raise RuntimeError(f"Number of items in results {funcs.config_names}"
                                       f" for should match the number of configs")

                for i, k in enumerate(funcs.config_names):
                    self.__set_configs(k, value[i], is_direct=False)

    def __compute_values(self):
        for k in self.topological_order:
            if k not in self.is_computed:
                self.__compute(k)

        parents = set()
        for k in self.topological_order:
            if k in self.aggregate_parent:
                parent = self.aggregate_parent[k]
                if parent not in self.is_computed:
                    parents.add(parent)

        self.topological_order = list(parents) + self.topological_order

    def __call__(self, run_order: Optional[List[Union[List[str], str]]]):
        if run_order is None:
            run_order = [list(self.types.keys())]
        run_order: List[Union[List[str], str]]

        for i in range(len(run_order)):
            keys = run_order[i]
            if type(keys) == str:
                run_order[i] = [keys]

        s = monit.section('Calculate config dependencies', is_new_line=False)
        with s:
            self.__create_graph()
        if s.get_estimated_time() >= 0.01:
            logger.log()

        self.visited = set()
        self.stack = []
        self.is_top_sorted = set()
        self.topological_order = []
        self.is_computed = set()

        for keys in run_order:
            self.__topological_sort(keys)
            self.__compute_values()
