import warnings
from typing import Tuple


class DynamicSchedule:
    def __init__(self, default: float, range_: Tuple[float, float] = (0, 1)):
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
        }
