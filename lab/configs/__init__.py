import ast
import inspect
import random
import string
import textwrap
import warnings
from collections import OrderedDict
from enum import Enum
from pathlib import PurePath
from typing import List, Dict, Callable, Type, cast, Set, Optional, \
    OrderedDict as OrderedDictType, Union, Any, Tuple

from lab import util


def random_string(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


_CALCULATORS = '_calculators'


class FunctionKind(Enum):
    pass_configs = 'pass_configs'
    pass_parameters = 'pass_parameters'


class Calculator:
    func: Callable
    kind: FunctionKind
    dependencies: Set[str]
    config_names: Union[str, List[str]]
    option_name: str
    is_append: bool
    params: List[inspect.Parameter]

    def __get_type(self):
        key, pos = 0, 0

        for p in self.params:
            if p.kind == p.POSITIONAL_OR_KEYWORD:
                pos += 1
            elif p.kind == p.KEYWORD_ONLY:
                key += 1
            else:
                assert False, "Only positional or keyword only arguments should be accepted"

        if pos == 1:
            assert key == 0
            return FunctionKind.pass_configs
        else:
            warnings.warn("Use configs object, because it's easier to refactor, find usage etc",
                          FutureWarning)
            assert pos == 0
            return FunctionKind.pass_parameters

    def __get_dependencies(self):
        if self.kind == FunctionKind.pass_configs:
            parser = DependencyParser(self.func)
            assert not parser.is_referenced, f"{self.func.__name__} should only use attributes of configs"
            return parser.required
        else:
            return {p.name for p in self.params}

    def __get_option_name(self, option_name: str):
        if option_name is not None:
            return option_name
        else:
            return self.func.__name__

    def __get_config_names(self, config_names: Union[str, List[str]]):
        if config_names is None:
            return self.func.__name__
        elif type(config_names) == str:
            return config_names
        else:
            assert type(config_names) == list
            assert len(config_names) > 0
            return config_names

    def __get_params(self):
        func_type = type(self.func)

        if func_type == type:
            init_func = cast(object, self.func).__init__
            spec: inspect.Signature = inspect.signature(init_func)
            params: List[inspect.Parameter] = list(spec.parameters.values())
            assert len(params) > 0
            assert params[0].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            assert params[0].name == 'self'
            return params[1:]
        else:
            spec: inspect.Signature = inspect.signature(self.func)
            params: List[inspect.Parameter] = list(spec.parameters.values())
            return params

    def __init__(self, func, *,
                 config_names: Union[str, List[str]],
                 option_name: str,
                 is_append: bool):
        self.func = func
        self.config_names = self.__get_config_names(config_names)
        self.is_append = is_append
        assert not (self.is_append and len(self.config_names) > 1)
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
        else:
            kwargs = {p.name: configs.__getattribute__(p.name) for p in self.params}
            return self.func(**kwargs)


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
        return cls.calc(name, f"_{random_string()}", is_append=True)


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


class DependencyParser(ast.NodeVisitor):
    def __init__(self, func: Callable):
        if type(func) == type:
            func = cast(object, func).__init__
            spec: inspect.Signature = inspect.signature(func)
            params = spec.parameters
            assert len(params) == 2
            param: inspect.Parameter = params[list(params.keys())[1]]
            source = textwrap.dedent(inspect.getsource(func))

        else:
            spec: inspect.Signature = inspect.signature(func)
            params = spec.parameters
            assert len(params) == 1
            param: inspect.Parameter = params[list(params.keys())[0]]
            source = inspect.getsource(func)

        assert (param.kind == param.POSITIONAL_ONLY or
                param.kind == param.POSITIONAL_OR_KEYWORD)

        self.arg_name = param.name

        self.required = set()
        self.is_referenced = False

        parsed = ast.parse(source)
        self.visit(parsed)

    def visit_Attribute(self, node: ast.Attribute):
        while not isinstance(node.value, ast.Name):
            if not isinstance(node.value, ast.Attribute):
                return

            node = node.value

        if node.value.id != self.arg_name:
            return

        self.required.add(node.attr)

    # Only visits if not captured before
    def visit_Name(self, node: ast.Name):
        if node.id == self.arg_name:
            self.is_referenced = True
            print(f"Referenced {node.id} in {node.lineno}:{node.col_offset}")


def _get_dependencies(func: Callable) -> Set[str]:
    parser = DependencyParser(func)
    assert not parser.is_referenced, f"{func} should only use attributes of configs"
    return parser.required


class ConfigPropertyException(ValueError):
    def __init__(self, func: Callable):
        file = inspect.getfile(func)
        source = inspect.getsource(func)
        super().__init__(f"{file}\n{source}\n")


class MissingConfigException(ValueError):
    def __init__(self, func: Callable, arg: str):
        file = inspect.getfile(func)
        source = inspect.getsource(func)
        super().__init__(f"Missing: {arg}\n{file}\n{source}\n")


class CyclicDependencyException(ValueError):
    def __init__(self, key):
        super().__init__(f"Cyclic dependency: {key}")


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

        if key in self.visited:
            raise CyclicDependencyException(key)

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
