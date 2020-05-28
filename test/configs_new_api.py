from labml.configs import BaseConfigs, option, calculate
from labml.internal.configs.processor import ConfigProcessor


class Sample(BaseConfigs):
    total_global_steps: int = 10
    workers_count: int = 12
    input_model: int
    model: int


@option(Sample.input_model)
def input_model(c: Sample):
    return c.total_global_steps * 2


calculate(Sample.model, [Sample.workers_count], lambda x: x * 5)

configs = Sample()

processor = ConfigProcessor(configs)
processor()
processor.print()

print(configs.__dict__)
