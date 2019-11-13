import functools
import inspect
import time

import typing


def option(config_name: str, option_name: str):
    def wrapper(func):
        @functools.wraps(func)
        def wrapper_func(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper_func.inner_func = func
        wrapper_func.config_name = config_name
        wrapper_func.option_name = option_name

        return wrapper_func

    return wrapper


class Configs:
    total_global_steps: int = 10
    workers_count: int = 10
    empty: str

    def input_model(self, *, workers_count: int):
        return workers_count * 2

    model = 'simple_model'

    @option('model', 'simple_model')
    def _model_simple_model(self, total_global_steps):
        return total_global_steps * 3


class Configs2(Configs):
    new_attr = 2


configs = Configs()

print(typing.get_type_hints(configs), flush=True)
print(typing.get_type_hints(Configs), flush=True)

print(typing.get_type_hints(configs.input_model), flush=True)
print(configs.__dict__, Configs.__dict__, flush=True)

print(Configs.__dict__['_model_simple_model'].__dict__, flush=True)

time.sleep(1)
print(inspect.getfullargspec(Configs.input_model), flush=True)

print(Configs2.__dict__, Configs2.__bases__)
print(isinstance(Configs2(), Configs))
