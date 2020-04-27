import inspect
from pathlib import PurePath
from typing import List, Dict, Callable, Optional, Union, Tuple

from .. import util
from .calculator import Calculator
from .config_function import ConfigFunction
from .config_item import ConfigItem
from .config_item import ConfigItem
from .parser import Parser, PropertyKeys
from ... import logger
from ...logger import Text

_CONFIG_PRINT_LEN = 40


def _is_class_method(func: Callable):
    if not callable(func):
        return False

    spec: inspect.Signature = inspect.signature(func)
    params: List[inspect.Parameter] = list(spec.parameters.values())
    if len(params) != 1:
        return False
    p = params[0]
    if p.kind != p.POSITIONAL_OR_KEYWORD:
        return False

    return p.name == 'self'


class Configs:
    r"""
    You should sub-class this class to create your own configurations
    """

    _calculators: Dict[str, List[ConfigFunction]] = {}
    _evaluators: Dict[str, List[ConfigFunction]] = {}

    def __init_subclass__(cls, **kwargs):
        configs = {}

        for k, v in cls.__annotations__.items():
            if not Parser.is_valid(k):
                continue

            configs[k] = ConfigItem(k,
                                    True, v,
                                    k in cls.__dict__, cls.__dict__.get(k, None))

        evals = []
        for k, v in cls.__dict__.items():
            if not Parser.is_valid(k):
                continue

            if _is_class_method(v):
                evals.append((k, v))
                continue

            configs[k] = ConfigItem(k,
                                    k in cls.__annotations__, cls.__annotations__.get(k, None),
                                    True, v)

        for e in evals:
            cls._add_eval_function(e[1], e[0], 'default')

        for k, v in configs.items():
            setattr(cls, k, v)

    @classmethod
    def _add_config_function(cls,
                             func: Callable,
                             name: Union[ConfigItem, List[ConfigItem]],
                             option: str, *,
                             is_append: bool
                             ):
        if PropertyKeys.calculators not in cls.__dict__:
            cls._calculators = {}

        calc = ConfigFunction(func, config_names=name, option_name=option, is_append=is_append)
        if type(calc.config_names) == str:
            config_names = [calc.config_names]
        else:
            config_names = calc.config_names

        for n in config_names:
            if n not in cls._calculators:
                cls._calculators[n] = []
            cls._calculators[n].append(calc)

    @classmethod
    def _add_eval_function(cls,
                           func: Callable,
                           name: str,
                           option: str):
        if PropertyKeys.evaluators not in cls.__dict__:
            cls._evaluators = {}

        calc = ConfigFunction(func,
                              config_names=name,
                              option_name=option,
                              is_append=False,
                              check_string_names=False)

        if name not in cls._evaluators:
            cls._evaluators[name] = []
        cls._evaluators[name].append(calc)

    @classmethod
    def calc(cls, name: Union[ConfigItem, List[ConfigItem]] = None,
             option: str = None, *,
             is_append: bool = False):
        r"""
        Use this as a decorator to register configuration options.

        Arguments:
            name: the configuration item or a list of items.
                If it is a list of items the function should return
                tuple.
            option (str, optional): name of the option.
                If not provided it will be derived from the
                function name.
        """

        def wrapper(func: Callable):
            cls._add_config_function(func, name, option, is_append=is_append)

            return func

        return wrapper

    @classmethod
    def list(cls, name: str = None):
        return cls.calc(name, f"_{util.random_string()}", is_append=True)

    @classmethod
    def set_hyperparams(cls, *args: ConfigItem, is_hyperparam=True):
        r"""
        Identifies configuration as (or not) hyper-parameters

        Arguments:
            *args: list of configurations
            is_hyperparam (bool, optional): whether the provided configuration
                items are hyper-parameters. Defaults to ``True``.
        """
        if PropertyKeys.hyperparams not in cls.__dict__:
            cls._hyperparams = {}

        for h in args:
            cls._hyperparams[h.key] = is_hyperparam

    @classmethod
    def aggregate(cls, name: Union[ConfigItem, any], option: str,
                  *args: Tuple[Union[ConfigItem, any], str]):
        r"""
        Aggregate configs

        Arguments:
            name: name of the aggregate
            option: aggregate option
            *args: list of options
        """

        assert args

        if PropertyKeys.aggregates not in cls.__dict__:
            cls._aggregates = {}

        if name.key not in cls._aggregates:
            cls._aggregates[name.key] = {}

        pairs = {p[0].key: p[1] for p in args}
        cls._aggregates[name.key][option] = pairs


class ConfigProcessor:
    def __init__(self, configs, values: Dict[str, any] = None):
        self.parser = Parser(configs, values)
        self.calculator = Calculator(configs=configs,
                                     options=self.parser.options,
                                     evals=self.parser.evals,
                                     types=self.parser.types,
                                     values=self.parser.values,
                                     list_appends=self.parser.list_appends,
                                     aggregate_parent=self.parser.aggregate_parent)

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
            return ConfigProcessor.__to_str(value)

    @staticmethod
    def __to_str(value):
        if str(value) == ConfigProcessor.__default_repr(value):
            if value.__class__.__module__ == '__main__':
                return value.__class__.__name__
            else:
                return f"{value.__class__.__module__}.{value.__class__.__name__}"
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
                'computed': self.__to_yaml(getattr(self.calculator.configs, k, None)),
                'is_hyperparam': self.parser.hyperparams.get(k, None),
                'is_explicitly_specified': (k in self.parser.explicitly_specified)
            }

        with open(str(configs_path), "w") as file:
            file.write(util.yaml_dump(configs))

    @staticmethod
    def __default_repr(value):
        return '<%s.%s object at %s>' % (
            value.__class__.__module__,
            value.__class__.__name__,
            hex(id(value))
        )

    def get_hyperparams(self):
        order = self.calculator.topological_order.copy()

        hyperparams = {}
        for key in order:
            if (self.parser.hyperparams.get(key, False) or
                    key in self.parser.explicitly_specified):
                value = getattr(self.calculator.configs, key, None)
                if key in self.parser.options:
                    value = self.parser.values[key]

                if type(value) not in {int, float, str}:
                    value = ConfigProcessor.__to_str(value)

                hyperparams[key] = value

        return hyperparams

    def __print_config(self, key, *, value=None, option=None,
                       other_options=None, is_ignored=False, is_list=False):
        parts = ['\t']

        if is_ignored:
            parts.append((key, Text.subtle))
            return parts

        is_hyperparam = self.parser.hyperparams.get(key, None)
        if is_hyperparam is None:
            is_hyperparam = key in self.parser.explicitly_specified
        if is_hyperparam:
            parts.append((key, [Text.key, Text.highlight]))
        else:
            parts.append((key, Text.key))

        if is_list:
            parts.append(('[]', Text.subtle))

        parts.append((' = ', Text.subtle))

        if other_options is None:
            other_options = []

        if value is not None:
            value_str = ConfigProcessor.__to_str(value)

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

        if value is None and option is None:
            parts.append(("None", Text.value))
            parts.append('\t')

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
        order.sort()
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

        logger.log()
