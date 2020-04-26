import ast
import inspect
import textwrap
import warnings
from enum import Enum
from typing import List, Callable, cast, Set, Union
from typing import TYPE_CHECKING

from .config_item import ConfigItem

if TYPE_CHECKING:
    from . import Configs


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

        source = textwrap.dedent(source)
        parsed = ast.parse(source)
        self.visit(parsed)

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Name):
            if node.value.id == self.arg_name:
                self.required.add(node.attr)
        else:
            for child in ast.iter_child_nodes(node):
                self.visit(child)

    def visit_Name(self, node: ast.Name):
        if node.id == self.arg_name:
            self.is_referenced = True
            print(f"Referenced {node.id} in {node.lineno}:{node.col_offset}")


def _get_dependencies(func: Callable) -> Set[str]:
    parser = DependencyParser(func)
    assert not parser.is_referenced, f"{func} should only use attributes of configs"
    return parser.required


class FunctionKind(Enum):
    pass_configs = 'pass_configs'
    pass_parameters = 'pass_parameters'
    pass_nothing = 'pass_nothing'


class ConfigFunction:
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
        elif pos == 0 and key == 0:
            return FunctionKind.pass_nothing
        else:
            warnings.warn("Use configs object, because it's easier to refactor, find usage etc",
                          FutureWarning, stacklevel=4)
            assert pos == 0
            return FunctionKind.pass_parameters

    def __get_dependencies(self):
        if self.kind == FunctionKind.pass_configs:
            parser = DependencyParser(self.func)
            assert not parser.is_referenced, \
                f"{self.func.__name__} should only use attributes of configs"
            return parser.required
        else:
            return {p.name for p in self.params}

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
                 is_append: bool,
                 check_string_names: bool = True):
        self.func = func
        self.check_string_names = check_string_names
        self.config_names = self.__get_config_names(config_names)
        self.is_append = is_append
        assert not (self.is_append and type(self.config_names) != str)
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
        elif self.kind == FunctionKind.pass_parameters:
            kwargs = {p.name: configs.__getattribute__(p.name) for p in self.params}
            return self.func(**kwargs)
        else:
            return self.func()
