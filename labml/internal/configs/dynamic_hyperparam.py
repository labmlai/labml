from typing import Tuple, Union


class DynamicHyperParam:
    def __init__(self, default: Union[float, int], type_: str, range_: Tuple[float, float]):
        self._type_ = type_
        self._range_ = range_
        self._default = default
        self._value = default
        self._registered = False

    def __call__(self):
        # if not self._registered:
        #     warnings.warn('Register dynamic schedules with `experiment.configs` to update them live from the app')
        return self._value

    def register(self):
        self._registered = True

    def set_value(self, value):
        self._value = value

    def to_yaml(self):
        return {
            'type': 'DynamicSchedule',
            'default': self._default,
            'range': list(self._range_),
            'dynamic_type': self._type_
        }


class FloatDynamicHyperParam(DynamicHyperParam):
    def __init__(self, default: float, range_: Tuple[float, float] = (0, 1)):
        super().__init__(default, 'float', range_)


class IntDynamicHyperParam(DynamicHyperParam):
    def __init__(self, default: int, range_: Tuple[int, int] = (1, 16)):
        super().__init__(default, 'int', range_)

    def set_value(self, value):
        self._value = int(value)
