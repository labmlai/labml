from pathlib import PurePath
from typing import List, Dict, Callable, Type, Optional, \
    Union

from lab import util, logger
from .calculator import Calculator
from .config_function import ConfigFunction
from .parser import Parser
from ..logger.colors import Text

_CALCULATORS = '_calculators'
_CONFIG_PRINT_LEN = 40


class Configs:
    _calculators: Dict[str, List[ConfigFunction]] = {}

    @classmethod
    def calc(cls, name: Union[str, List[str]] = None,
             option: str = None, *,
             is_append: bool = False):
        if _CALCULATORS not in cls.__dict__:
            cls._calculators = {}

        def wrapper(func: Callable):

            calc = ConfigFunction(func, config_names=name, option_name=option, is_append=is_append)
            if type(calc.config_names) == str:
                config_names = [calc.config_names]
            else:
                config_names = calc.config_names

            for n in config_names:
                if n not in cls._calculators:
                    cls._calculators[n] = []
                cls._calculators[n].append(calc)

            return func

        return wrapper

    @classmethod
    def list(cls, name: str = None):
        return cls.calc(name, f"_{util.random_string()}", is_append=True)


def _get_base_classes(class_: Type[Configs]) -> List[Type[Configs]]:
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


class ConfigProcessor:
    def __init__(self, configs, values: Dict[str, any] = None):
        self.parser = Parser(configs, values)
        self.calculator = Calculator(configs=configs,
                                     options=self.parser.options,
                                     types=self.parser.types,
                                     values=self.parser.values,
                                     list_appends=self.parser.list_appends)

    def __call__(self, run_order: Optional[List[Union[List[str], str]]] = None):
        self.calculator(run_order)

    @staticmethod
    def __is_primitive(value):
        if value is None:
            return True

        if type(value) == str:
            return True

        if type(value) == int:
            return True

        if type(value) == bool:
            return True

        if type(value) == list and all([ConfigProcessor.__is_primitive(v) for v in value]):
            return True

        if type(value) == dict and all([ConfigProcessor.__is_primitive(v) for v in value.values()]):
            return True

        return False

    @staticmethod
    def __to_yaml(value):
        if ConfigProcessor.__is_primitive(value):
            return value
        else:
            return str(value)

    def save(self, configs_path: PurePath):
        orders = {k: i for i, k in enumerate(self.calculator.topological_order)}
        configs = {}
        for k, v in self.parser.types.items():
            configs[k] = {
                'name': k,
                'type': str(v),
                'value': self.__to_yaml(self.parser.values.get(k, None)),
                'order': orders.get(k, -1),
                'options': list(self.parser.options.get(k, {}).keys()),
                'computed': self.__to_yaml(getattr(self.calculator.configs, k, None))
            }

        with open(str(configs_path), "w") as file:
            file.write(util.yaml_dump(configs))

    @staticmethod
    def __print_config(key, *, value=None, option=None,
                       other_options=None, is_ignored=False, is_list=False):
        parts = ['\t']

        if is_ignored:
            parts.append((key, Text.subtle))
            return parts

        parts.append((key, Text.key))

        if is_list:
            parts.append(('[]', Text.subtle))

        parts.append((' = ', Text.subtle))

        if other_options is None:
            other_options = []

        if value is not None:
            value_str = str(value)
            value_str = value_str.replace('\n', '')
            if len(value_str) < _CONFIG_PRINT_LEN:
                parts.append((f"{value_str}", Text.value))
            else:
                parts.append((f"{value_str[:_CONFIG_PRINT_LEN]}...", Text.value))
            parts.append('\t')

        if option is not None:
            if len(other_options) == 0:
                parts.append((option, Text.subtle))
            else:
                parts.append((option, Text.none))

        if len(other_options) > 0:
            parts.append(('\t[', Text.subtle))
            for i, opt in enumerate(other_options):
                if i > 0:
                    parts.append((', ', Text.subtle))
                parts.append(opt)
            parts.append((']', Text.subtle))

        return parts

    def print(self):
        order = self.calculator.topological_order.copy()
        added = set(order)
        ignored = set()

        for k in self.parser.types:
            if k not in added:
                added.add(k)
                order.append(k)
                ignored.add(k)

        logger.log("Configs:", Text.heading)

        for k in order:
            computed = getattr(self.calculator.configs, k, None)

            if k in ignored:
                parts = self.__print_config(k, is_ignored=True)
            elif k in self.parser.list_appends:
                parts = self.__print_config(k,
                                            value=computed,
                                            is_list=True)
            elif k in self.parser.options:
                v = self.parser.values[k]
                opts = self.parser.options[k]
                lst = list(opts.keys())
                if v in opts:
                    lst.remove(v)
                else:
                    v = None

                parts = self.__print_config(k,
                                            value=computed,
                                            option=v,
                                            other_options=lst)
            else:
                parts = self.__print_config(k, value=computed)

            logger.log(parts)

        logger.new_line()
