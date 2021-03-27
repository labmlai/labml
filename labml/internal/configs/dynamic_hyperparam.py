import warnings
from typing import Tuple, Union

from labml.logger import Text

from labml import tracker, logger


class DynamicHyperParam:
    def __init__(self, default: Union[float, int], type_: str, range_: Tuple[float, float]):
        self._type_ = type_
        self._range_ = range_
        self._default = default
        self._value = default
        self._key = None
        self._change_tracked = False
        self._last_tracked_step = -1

    def __call__(self):
        if not self._change_tracked or self._last_tracked_step != tracker.get_global_step():
            if self._key is None:
                warnings.warn('Register dynamic schedules with `experiment.configs` to update them live from the app')
            else:
                tracker.add(f'hp.{self._key}', self._value)

        self._change_tracked = True
        self._last_tracked_step = tracker.get_global_step()

        return self._value

    def register(self, key):
        self._key = key

    def set_value(self, value):
        if self._key is None:
            warnings.warn('Register dynamic schedules with `experiment.configs` to update them live from the app')
        else:
            logger.log([
                ('Dynamic hyper parameter changed: ', Text.subtle),
                (self._key, Text.key),
                (': ', Text.subtle),
                (f'{self._value}', Text.value),
                (' -> ', Text.subtle),
                (f'{value}', Text.value),
            ])
        self._change_tracked = False
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
        super().set_value(int(value))
