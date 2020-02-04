from collections import OrderedDict
from typing import List, Dict, Type, OrderedDict as OrderedDictType, Any
from typing import TYPE_CHECKING

from .config_function import ConfigFunction
from .config_item import ConfigItem

if TYPE_CHECKING:
    from . import Configs

_CALCULATORS = '_calculators'


def _get_base_classes(class_: Type['Configs']) -> List[Type['Configs']]:
    classes = [class_]
    level = [class_]
    next_level = []

    while len(level) > 0:
        for c in level:
            for b in c.__bases__:
                if b == object:
                    continue
                next_level.append(b)
        classes += next_level
        level = next_level
        next_level = []

    classes.reverse()

    return classes


RESERVED = {'calc', 'list'}
_STANDARD_TYPES = {int, str, bool, Dict, List}


class Parser:
    config_items: Dict[str, ConfigItem]
    options: Dict[str, OrderedDictType[str, ConfigFunction]]
    types: Dict[str, Type]
    values: Dict[str, any]
    list_appends: Dict[str, List[ConfigFunction]]

    def __init__(self, configs: 'Configs', values: Dict[str, any] = None):
        classes = _get_base_classes(type(configs))

        self.values = {}
        self.types = {}
        self.options = {}
        self.list_appends = {}
        self.config_items = {}
        self.configs = configs

        for c in classes:
            # for k, v in c.__annotations__.items():
            #     self.__collect_annotation(k, v)
            #
            for k, v in c.__dict__.items():
                self.__collect_config_item(k, v)

        for c in classes:
            if _CALCULATORS in c.__dict__:
                for k, calcs in c.__dict__[_CALCULATORS].items():
                    assert k in self.types, \
                        f"{k} calculator is present but the config declaration is missing"
                    for v in calcs:
                        self.__collect_calculator(k, v)

        for k, v in configs.__dict__.items():
            assert k in self.types
            self.__collect_value(k, v)

        if values is not None:
            for k, v in values.items():
                assert k in self.types
                self.__collect_value(k, v)

        self.__calculate_missing_values()

    @staticmethod
    def is_valid(key):
        if key.startswith('_'):
            return False

        if key in RESERVED:
            return False

        return True

    def __collect_config_item(self, k, v: ConfigItem):
        if not self.is_valid(k):
            return

        if v.has_value:
            self.values[k] = v.value

        if k in self.config_items:
            self.config_items[k].update(v)
        else:
            self.config_items[k] = v

        if k not in self.types:
            self.types[k] = v.annotation

    def __collect_value(self, k, v):
        if not self.is_valid(k):
            return

        self.values[k] = v
        if k not in self.types:
            self.types[k] = type(v)

    def __collect_annotation(self, k, v):
        if not self.is_valid(k):
            return

        self.types[k] = v

    def __collect_calculator(self, k, v: ConfigFunction):
        if v.is_append:
            if k not in self.list_appends:
                self.list_appends[k] = []
            self.list_appends[k].append(v)
        else:
            if k not in self.options:
                self.options[k] = OrderedDict()
            if v.option_name in self.options[k]:
                assert v == self.options[k][
                    v.option_name], f"Duplicate option for {k}: {v.option_name}"

            self.options[k][v.option_name] = v

    def __calculate_missing_values(self):
        for k in self.types:
            if k in self.values and self.values[k] is not None:
                continue

            if k in self.list_appends:
                continue

            if k in self.options:
                self.values[k] = next(iter(self.options[k].keys()))
                continue

            if type(self.types[k]) == type:
                if self.types[k] in _STANDARD_TYPES:
                    continue

                self.options[k] = OrderedDict()
                self.options[k][k] = ConfigFunction(self.types[k],
                                                    config_names=self.config_items[k],
                                                    option_name=k,
                                                    is_append=False)
                self.values[k] = k
                continue

            assert k in self.values, f"Cannot compute {k}"
