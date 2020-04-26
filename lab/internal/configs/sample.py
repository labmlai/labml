from typing import List

from lab.internal.configs import Configs, ConfigProcessor


class SampleModel:
    def __init__(self, c: 'Sample'):
        self.w = c.workers_count


class Sample(Configs):
    total_global_steps: int = 10
    workers_count: int = 10
    # empty: str

    x = 'string'

    input_model: int
    model: int

    # get from type annotations
    model_obj: SampleModel

    steps: List[any]


class SampleChild(Sample):
    def __init__(self, *, test: int):
        pass

    new_attr = 2


@Sample.calc()
def input_model(c: Sample):
    return c.workers_count * 2


@Sample.calc(Sample.input_model)
def input_model2(c: Sample):
    return c.workers_count * 20


@Sample.calc('model')
def simple_model(c: Sample):
    return c.total_global_steps * 3


# When collecting unordered items
@Sample.list('steps')
def remove_first():
    return None


@Sample.list('steps')
def model_step(c: Sample):
    return c.model


configs = Sample()

processor = ConfigProcessor(configs)
processor()

print(configs.__dict__)
