from collections import OrderedDict
from typing import Dict, Any, OrderedDict as OrderedDictType


class HParameter:
    _value: OrderedDictType[str, Any]

    def __init__(self, name: str, is_print: bool):
        self.name = name
        self.is_print = is_print

        self._value = OrderedDict()

    def get_value(self):
        return self._value

    def collect_value(self, value: Dict):
        self._value = value

    def clear(self):
        self._value = OrderedDict()

    def is_empty(self):
        return len(self._value) == 0
