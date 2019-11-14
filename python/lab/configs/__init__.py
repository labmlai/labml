import inspect
import typing
from typing import List, NamedTuple, Dict


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


class CyclicDependencyException(ValueError):
    def __init__(self, key):
        super().__init__(f"Cyclic dependency: {key}")


class ConfigProcessor:
    options: Dict[str, Dict[str, Option]]
    list_appends: Dict[str, List[ListAppend]]
    properties: Dict[str, typing.Callable]

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
        if k.startswith('__') and k.endswith('__'):
            return

        if type(v) == Option:
            v = typing.cast(Option, v)
            key = v.config_name
            if key not in self.options:
                self.options[key] = {}
            self.options[key][v.option_name] = v
        elif type(v) == ListAppend:
            v = typing.cast(ListAppend, v)
            key = v.list_name
            if key not in self.list_appends:
                self.list_appends[key] = []
            self.list_appends[key].append(v)
        elif callable(v):
            self.properties[k] = v
        elif type(v) == staticmethod:
            self.properties[k] = v.__func__
        else:
            self.values[k] = v

    def __check_properties(self):
        """
        Checks all the `properties`, `options` and `list_appends`
        """

        for p in self.properties.values():
            self.__check_callable(p)
        for opts in self.options.values():
            for p in opts.values():
                self.__check_callable(p.func)
        for appends in self.list_appends.values():
            for p in appends:
                self.__check_callable(p.func)

    def __check_callable(self, func: typing.Callable):
        arg_spec = inspect.getfullargspec(func)
        func_type = type(func)

        if func_type == type:
            if arg_spec.args != ['self']:
                raise ConfigPropertyException(func)
        else:
            if arg_spec.args != ['self'] and arg_spec.args != []:
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
            if ((arg not in self.values) and (arg not in self.properties) and
                    (arg not in self.list_appends)):
                raise MissingConfigException(func, arg)

    def __get_property(self, key):
        if key in self.properties:
            assert key not in self.values
            assert key not in self.list_appends
            assert key not in self.options

            return None, self.properties[key]
        elif key in self.values:
            assert key not in self.properties
            assert key not in self.list_appends
            if key in self.options:
                return None, self.options[key][self.values[key]].func
            else:
                return self.values[key], None
        elif key in self.list_appends:
            assert key not in self.options

            return None, [v.func for v in self.list_appends[key]]
        else:
            raise ValueError(f"Missing config {key}")

    def __add_to_topological_order(self, key):
        assert self.stack.pop() == key
        self.is_computed.add(key)
        self.topological_order.append(key)

    def __get_args(self, func: typing.Callable):
        arg_spec = inspect.getfullargspec(func)
        return arg_spec.args, arg_spec.kwonlyargs

    def __traverse(self, key):
        value, funcs = self.__get_property(key)
        if value is not None:
            self.__add_to_topological_order(key)
            return

        if funcs is None:
            assert False

        if type(funcs) != list:
            funcs = [funcs]

        for f in funcs:
            _, args = self.__get_args(f)
            for a in args:
                if a not in self.is_computed:
                    self.__add_to_stack(a)
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

    def __topological_sort(self):
        self.visited = set()
        self.stack = []
        self.is_computed = set()
        self.topological_order = []

        for k in self.values:
            self.__add_to_stack(k)
            self.__dfs()

        for k in self.properties:
            self.__add_to_stack(k)
            self.__dfs()

        for k in self.list_appends:
            self.__add_to_stack(k)
            self.__dfs()

    def __call_func(self, func: typing.Callable):
        s_, args = self.__get_args(func)
        kwargs = {k: self.computed[k] for k in args}

        if type(func) == type:
            return func(**kwargs)
        else:
            if len(s_) > 0:
                return func(None, **kwargs)
            else:
                return func(**kwargs)

    def __compute(self, key):
        value, funcs = self.__get_property(key)
        if value is not None:
            return value
        elif type(funcs) == list:
            return [self.__call_func(f) for f in funcs]
        else:
            return self.__call_func(funcs)

    def __compute_values(self):
        self.computed = {}

        for k in self.topological_order:
            self.computed[k] = self.__compute(k)

    def calculate(self):
        self.__check_properties()
        self.__topological_sort()
        self.__compute_values()
        print(self.computed)
