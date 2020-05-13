import inspect
from typing import List, Callable, Set

from labml.internal.configs.dependency_parser import DependencyParser


class EvalFunction:
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
                assert False, "Only positional or keyword only arguments should be accepted"

        assert pos >= 1

    def __get_dependencies(self):
        parser = DependencyParser(self.func)
        if parser.is_referenced:
            raise RuntimeError(f"{self.func.__name__} should only use attributes of configs")
        return parser.required

    def __init__(self, func, *,
                 config_name: str):
        self.func = func
        self.config_name = config_name

        self.__check_type()
        self.dependencies = self.__get_dependencies()
