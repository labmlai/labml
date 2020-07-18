from labml.configs import option
from labml.internal.configs.base import Configs
from labml.internal.configs.processor import ConfigProcessor


class SampleModel:
    def __init__(self, prop1: int, prop2: int):
        self.prop2 = prop2
        self.prop1 = prop1


class SampleConfigsModule(Configs):
    prop1: int
    prop2: int = 10
    # prop3: int
    model: SampleModel


class SampleConfigs(Configs):
    my_prop1: int = 2
    module: SampleConfigsModule
    run_model: SampleModel


@option(SampleConfigsModule.model)
def sample_model(c: SampleConfigsModule):
    return SampleModel(c.prop1, c.prop2)


@option(SampleConfigs.module)
def sample_configs_module(c: SampleConfigs):
    conf = SampleConfigsModule()
    conf.prop1 = c.my_prop1
    return conf


@option(SampleConfigs.run_model)
def run_model(c: SampleConfigs):
    print((2 + 2).__str__())
    return c.module.model


def test():
    configs = SampleConfigs()

    processor = ConfigProcessor(configs)
    processor()
    processor.print()

    print(configs.__dict__)
    print(configs.module.__dict__)


if __name__ == '__main__':
    test()
