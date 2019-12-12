import ast
import inspect
import random
import string
import textwrap
from collections import OrderedDict
from enum import Enum
from typing import List, NamedTuple, Dict, Callable, Type, cast, Set, Optional, \
    OrderedDict as OrderedDictType, Union


def random_string(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


class Calculator(NamedTuple):
    func: Callable
    config_name: str
    option_name: str
    is_append: bool


_CALCULATORS = '_calculators'


class Configs:
    _calculators: Dict[str, List[Calculator]] = {}

    @classmethod
    def calc(cls, name: Union[str, List[str]] = None, option: str = None, *,
             is_append: bool = None):
        if _CALCULATORS not in cls.__dict__:
            cls._calculators = {}

        def wrapper(func: Callable):
            _name = func.__name__ if name is None else name
            if option is not None:
                _option = option
            else:
                _option = func.__name__

            calc = Calculator(func, _name, option, is_append)
            if _name not in cls._calculators:
                cls._calculators[_name] = []

            cls._calculators[_name].append(calc)

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


class FunctionType(Enum):
    pass_configs = 'pass_configs'
    pass_parameters = 'pass_parameters'


class Function:
    func: Callable
    type_: FunctionType
    dependencies: Set[str]
    results: List[str]

    def __get_type(self):
        return FunctionType.pass_configs

    def __get_dependencies(self):
        return set()

    def __init__(self, func, results):
        self.func = func
        self.type_ = self.__get_type()
        self.dependencies = self.__get_dependencies()
        self.results = results

    def __call__(self, configs: Configs):
        # Call and set values
        pass


class ConfigProcessor:
    options: Dict[str, OrderedDictType[str, Callable]]
    types: Dict[str, Type]
    values: Dict[str, any]
    list_appends: Dict[str, List[Callable]]
    dependencies: Dict[str, Set[str]]

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

            if _CALCULATORS in c.__dict__:
                for k, calcs in c.__dict__['_calculators'].items():
                    for v in calcs:
                        self.__collect_calculator(k, v)

        for k, v in configs.__dict__.items():
            self.__collect_value(k, v)

        if values is not None:
            for k, v in values.items():
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
            self.list_appends[k].append(v.func)
        else:
            if k not in self.options:
                self.options[k] = OrderedDict()
            self.options[k][v.option_name] = v.func

    def __check_properties(self):
        """
        Checks all the `properties`, `options` and `list_appends`
        """

        for opts in self.options.values():
            for p in opts.values():
                self.__check_callable(p)
        for appends in self.list_appends.values():
            for p in appends:
                self.__check_callable(p)

    def __check_callable(self, func: Callable):
        func_type = type(func)

        if func_type == type:
            init_func = cast(object, func).__init__
            spec: inspect.Signature = inspect.signature(init_func)
            params = spec.parameters
            if len(params) != 2 and len(params) != 1:
                raise ConfigPropertyException(func)
        else:
            spec: inspect.Signature = inspect.signature(func)
            params = spec.parameters
            if len(params) > 1:
                raise ConfigPropertyException(func)

    def __call_func(self, func: Callable):
        func_type = type(func)

        if func_type == type:
            init_func = cast(object, func).__init__
            spec: inspect.Signature = inspect.signature(init_func)
            params = spec.parameters
            if len(params) == 2:
                return func(self.configs)
            else:
                return func()
        else:
            spec: inspect.Signature = inspect.signature(func)
            params = spec.parameters
            if len(params) == 1:
                return func(self.configs)
            else:
                return func()

    def __get_property(self, key):
        if key in self.options:
            value = self.values[key]
            return None, self.options[key][value]

        if key in self.list_appends:
            return None, [f for f in self.list_appends[key]]

        return self.values[key], None

    def __add_to_topological_order(self, key):
        assert self.stack.pop() == key
        self.is_computed.add(key)
        self.topological_order.append(key)

    def __traverse(self, key):
        for d in self.dependencies[key]:
            if d not in self.is_computed:
                self.__add_to_stack(d)
                return

        self.__add_to_topological_order(key)

    def __add_to_stack(self, key):
        if key in self.is_computed:
            return

        if key in self.visited:
            raise CyclicDependencyException(key)

        self.visited.add(key)
        self.stack.append(key)

    def __dfs(self):
        while len(self.stack) > 0:
            key = self.stack[-1]
            self.__traverse(key)

    def __calculate_missing_values(self):
        for k in self.types:
            if k in self.values:
                continue

            if k in self.list_appends:
                continue

            if k in self.options:
                self.values[k] = next(iter(self.options[k].keys()))
                continue

            if type(self.types[k]) == type:
                self.options[k] = OrderedDict()
                self.options[k][k] = self.types[k]
                self.values[k] = k
                continue

            assert False, f"Cannot compute {k}"

    def __get_dependencies(self, key) -> Set[str]:
        assert not (key in self.options and key in self.list_appends), \
            f"{key} in options and appends"

        if key in self.options:
            value = self.values[key]
            func = self.options[key][value]
            return _get_dependencies(func)

        if key in self.list_appends:
            dep = set()
            for func in self.list_appends[key]:
                dep = dep.union(_get_dependencies(func))

            return dep

        return set()

    def __create_graph(self):
        self.dependencies = {}
        for k in self.types:
            self.dependencies[k] = self.__get_dependencies(k)

    def __topological_sort(self, keys: Optional[List[str]] = None):
        self.visited = set()
        self.stack = []
        self.is_computed = set()
        self.topological_order = []

        if keys is None:
            keys = self.types.keys()

        for k in keys:
            self.__add_to_stack(k)
            self.__dfs()

    def __compute(self, key):
        value, funcs = self.__get_property(key)
        if value is not None:
            return value
        elif type(funcs) == list:
            return [self.__call_func(f) for f in funcs]
        else:
            return self.__call_func(funcs)

    def __compute_values(self):
        for k in self.topological_order:
            self.configs.__setattr__(k, self.__compute(k))

    def calculate(self):
        self.__check_properties()
        self.__calculate_missing_values()
        self.__create_graph()
        self.__topological_sort()
        self.__compute_values()

        print(self.configs)
