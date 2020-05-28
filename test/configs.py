from labml.internal.configs.base import Configs
from labml.internal.configs.processor import ConfigProcessor


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


configs = Sample()

processor = ConfigProcessor(configs)
processor()
processor.print()

print(configs.__dict__)
