from typing import Callable


class EvalFunction:
    func: Callable
    config_name: str

    def __init__(self, func, *,
                 config_name: str):
        self.func = func
        self.config_name = config_name
