import inspect
from typing import List, Callable


class EvalFunction:
    func: Callable
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
                raise RuntimeError(f"Only positional or keyword only arguments should be accepted: "
                                   f"{self.config_name} - {self.func.__name__}")

        assert pos >= 1

    def __init__(self, func, *,
                 config_name: str):
        self.func = func
        self.config_name = config_name

        self.__check_type()
