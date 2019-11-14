import functools
import inspect
import time

import typing
from lab.configs import option, append, Configs


class Sample(Configs):
    total_global_steps: int = 10
    workers_count: int = 10
    empty: str
    steps = []

    def instance_func(self, *, x:int):
        pass

    @staticmethod
    def input_model(*, workers_count: int):
        return workers_count * 2

    model = 'simple_model'

    @staticmethod
    @option('model', 'simple_model')
    def _model_simple_model(total_global_steps):
        return total_global_steps * 3

    # When collecting unordered items
    @append('steps')
    def remove_first(self):
        return None

    @append('steps')
    def model_step(self, model):
        return model


class SampleChild(Sample):
    def __init__(self, *, test: int):
        pass

    new_attr = 2


configs = Sample()

print(typing.get_type_hints(configs), flush=True)
print(typing.get_type_hints(Sample), flush=True)

print(typing.get_type_hints(configs.input_model), flush=True)
print(configs.__dict__, Sample.__dict__, flush=True)

# print(Sample.__dict__['_model_simple_model'].__dict__, flush=True)

time.sleep(1)
print(type(Sample.instance_func), inspect.getfullargspec(Sample.instance_func), flush=True)
print(type(Sample.input_model), inspect.getfullargspec(Sample.input_model), flush=True)

print(SampleChild.__dict__, SampleChild.__bases__)
print(isinstance(SampleChild(test=1), Sample))

print(Sample.__bases__)
print(type(SampleChild), inspect.getfullargspec(SampleChild))
print(inspect.getfile(SampleChild), inspect.getsource(SampleChild))
f = SampleChild
x = f(test=2)
