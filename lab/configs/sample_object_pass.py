import typing


class Configs:
    __computers = {}
    test = {}

    @classmethod
    def compute(cls, name=None, option_name=None, *, is_default=None, is_append=None, value=None):
        if cls.__name__ not in computers:
            computers[cls.__name__] = {}
        print(cls.test)
        if '_computers' not in cls.__dict__:
            cls.__computers = {}
            cls.__test = {}
            cls.test = 'name'

        def wrapper(func: typing.Callable):
            _name = func.__name__ if name is None else name
            _option_name = func.__name__ if option_name is None else option_name
            computers[cls.__name__][_name] = func
            cls.__computers[_name] = func

            return func

        if value is None:
            return wrapper
        else:
            return wrapper(value)


class SampleModel:
    def __init__(self, *, workers_count):
        self.w = workers_count


class Sample(Configs):
    x = 'string'
    total_global_steps: int = 10
    workers_count: int = 10
    empty: str

    input_model: int
    model: int

    # get from type annotations
    model_obj: SampleModel
    # def instance_func(self, *, x:int):
    #     pass


computers = {}


def compute(cls, name=None, option_name=None, *, is_default=None, is_append=None, value=None):
    if cls.__name__ not in computers:
        computers[cls.__name__] = {}

    def wrapper(func: typing.Callable):
        _name = func.__name__ if name is None else name
        _option_name = func.__name__ if option_name is None else option_name
        computers[cls.__name__][name] = func
        return func

    if value is None:
        return wrapper
    else:
        return wrapper(value)


@Sample.compute()
def input_model(c: Sample):
    return c.workers_count * 2


@compute(Sample, 'model')
def simple_model(self, *, total_global_steps):
    return total_global_steps * 3


# When collecting unordered items
@compute(Sample, 'steps')
def remove_first(self):
    return None


@compute(Sample, 'steps')
def model_step(self, *, model):
    return model


compute(Sample, 'model_obj', value=SampleModel)


@compute(Sample)
def print(c: Sample):
    print(c.model_obj, c.model_obj.w)


print(2)
