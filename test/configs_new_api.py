from labml.configs import BaseConfigs, option
from labml.internal.configs.processor import ConfigProcessor


class Sample(BaseConfigs):
    total_global_steps: int = 10
    workers_count: int = 10
    input_model: int
    # model: int


@option(Sample.input_model)
def input_model(c: Sample):
    return c.workers_count * 2


configs = Sample()

processor = ConfigProcessor(configs)
processor()
processor.print()

print(configs.__dict__)
