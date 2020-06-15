import warnings
from collections import OrderedDict
from typing import List, Dict, Type, OrderedDict as OrderedDictType, Set
from typing import TYPE_CHECKING

from labml import logger
from labml.internal.configs.eval_function import EvalFunction
from labml.logger import Text

from .config_function import ConfigFunction
from .config_item import ConfigItem

if TYPE_CHECKING:
    from .base import Configs

RESERVED = {'calc', 'list', 'set_hyperparams', 'aggregate', 'calc_wrap'}
_STANDARD_TYPES = {int, str, bool, Dict, List}


class PropertyKeys:
    calculators = '_calculators'
    evaluators = '_evaluators'
    hyperparams = '_hyperparams'
    aggregates = '_aggregates'


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

    unique_classes = []
    hashes: Set[int] = set()
    for c in classes:
        if hash(c) not in hashes:
            unique_classes.append(c)
        hashes.add(hash(c))

    return unique_classes


class Parser:
    config_items: Dict[str, ConfigItem]
    options: Dict[str, OrderedDictType[str, ConfigFunction]]
    evals: Dict[str, OrderedDictType[str, EvalFunction]]
    types: Dict[str, Type]
    values: Dict[str, any]
    explicitly_specified: Set[str]
    hyperparams: Dict[str, bool]
    aggregates: Dict[str, Dict[str, Dict[str, str]]]
    aggregate_parent: Dict[str, str]

    def __init__(self, configs: 'Configs', values: Dict[str, any] = None):
        classes = _get_base_classes(type(configs))

        self.values = {}
        self.types = {}
        self.options = {}
        self.evals = {}
        self.config_items = {}
        self.configs = configs
        self.explicitly_specified = set()
        self.hyperparams = {}
        self.aggregates = {}
        self.aggregate_parent = {}

        for c in classes:
            # for k, v in c.__annotations__.items():
            #     self.__collect_annotation(k, v)
            #
            for k, v in c.__dict__.items():
                if (PropertyKeys.evaluators in c.__dict__ and
                        k in c.__dict__[PropertyKeys.evaluators]):
                    continue
                self.__collect_config_item(k, v)

        for c in classes:
            if PropertyKeys.calculators in c.__dict__:
                for k, calculators in c.__dict__[PropertyKeys.calculators].items():
                    assert k in self.types, \
                        f"{k} calculator is present but the config declaration is missing"
                    for v in calculators:
                        self.__collect_calculator(k, v)

        for c in classes:
            if PropertyKeys.evaluators in c.__dict__:
                for k, evaluators in c.__dict__[PropertyKeys.evaluators].items():
                    for v in evaluators:
                        self.__collect_evaluator(k, v)

        for c in classes:
            if PropertyKeys.hyperparams in c.__dict__:
                for k, is_hyperparam in c.__dict__[PropertyKeys.hyperparams].items():
                    self.hyperparams[k] = is_hyperparam

        for c in classes:
            if PropertyKeys.aggregates in c.__dict__:
                for k, aggregates in c.__dict__[PropertyKeys.aggregates].items():
                    self.aggregates[k] = aggregates

        for k, v in configs.__dict__.items():
            assert k in self.types
            self.__collect_value(k, v)

        if values is not None:
            for k, v in values.items():
                if k in self.types:
                    self.__collect_value(k, v)
                else:
                    logger.log(f'Ignoring config: {k} = {str(v)}', Text.warning)

        self.__calculate_aggregates()
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

        self.explicitly_specified.add(k)

        self.values[k] = v
        if k not in self.types:
            self.types[k] = type(v)

    def __collect_annotation(self, k, v):
        if not self.is_valid(k):
            return

        self.types[k] = v

    def __collect_calculator(self, k, v: ConfigFunction):
        if k not in self.options:
            self.options[k] = OrderedDict()
        if v.option_name in self.options[k]:
            if v != self.options[k][v.option_name]:
                warnings.warn(f"Overriding option for {k}: {v.option_name}", Warning,
                              stacklevel=5)

        self.options[k][v.option_name] = v

    def __collect_evaluator(self, k, v: EvalFunction):
        if k not in self.evals:
            self.evals[k] = OrderedDict()

        self.evals[k]['default'] = v

    def __calculate_missing_values(self):
        for k in self.types:
            if k in self.values:
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
                                                    option_name=k)
                self.values[k] = k
                continue

            assert k in self.values, f"Cannot compute {k}"

    def __calculate_aggregates(self):
        queue = []
        for k in self.aggregates:
            if k not in self.values or self.values[k] not in self.aggregates[k]:
                continue

            queue.append(k)

        while queue:
            k = queue.pop()
            assert k in self.values
            option = self.values[k]
            assert option in self.aggregates[k]
            pairs = self.aggregates[k][option]

            for name, opt in pairs.items():
                if name in self.values and self.values[name] != '__none__':
                    continue

                self.aggregate_parent[name] = k
                self.values[name] = opt

                if name in self.aggregates and opt in self.aggregates[name]:
                    queue.append(name)
