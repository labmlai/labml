from labml.internal.configs2.base import Configs


class SampleModel:
    def __init__(self, prop1: int, prop2: int):
        self.prop2 = prop2
        self.prop1 = prop1

    def __str__(self):
        return f"{self.prop1}, {self.prop2}"


class SampleConfigsModule(Configs):
    prop1: int
    prop2: int = 20
    prop3: int = 10
    model: SampleModel


@SampleConfigsModule.calc(SampleConfigsModule.model)
def sample_model(c: SampleConfigsModule):
    return SampleModel(c.prop1, c.prop2)


def test():
    configs = SampleConfigsModule()
    configs.prop1 = 3
    print(configs.model)


if __name__ == '__main__':
    test()
