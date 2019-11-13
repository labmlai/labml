import functools
import inspect
import time

import typing
from lab.configs import option, append


class Sample:
    total_global_steps: int = 10
    workers_count: int = 10
    empty: str
    steps = []

    def input_model(self, *, workers_count: int):
        return workers_count * 2

    model = 'simple_model'

    @option('model', 'simple_model')
    def _model_simple_model(self, total_global_steps):
        return total_global_steps * 3

    # When collecting unordered items
    @append('steps')
    def remove_first(self):
        return None

    @append('steps')
    def model_step(self, model):
        return model


class SampleChild(Sample):
    new_attr = 2


configs = Sample()

print(typing.get_type_hints(configs), flush=True)
print(typing.get_type_hints(Sample), flush=True)

print(typing.get_type_hints(configs.input_model), flush=True)
print(configs.__dict__, Sample.__dict__, flush=True)

# print(Sample.__dict__['_model_simple_model'].__dict__, flush=True)

time.sleep(1)
print(inspect.getfullargspec(Sample.input_model), flush=True)

print(SampleChild.__dict__, SampleChild.__bases__)
print(isinstance(SampleChild(), Sample))

print(Sample.__bases__)
