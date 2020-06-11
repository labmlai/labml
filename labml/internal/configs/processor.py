from pathlib import PurePath, Path
from typing import Dict, Optional, List, Union

from labml import logger
from labml.internal import util
from labml.internal.configs.base import Configs
from labml.internal.configs.utils import Value
from labml.logger import Text

from .calculator import Calculator
from .parser import Parser

_CONFIG_PRINT_LEN = 40


class ConfigProcessor:
    def __init__(self, configs: Configs, values: Dict[str, any] = None):
        self.parser = Parser(configs, values)
        self.calculator = Calculator(configs=configs,
                                     options=self.parser.options,
                                     evals=self.parser.evals,
                                     types=self.parser.types,
                                     values=self.parser.values,
                                     aggregate_parent=self.parser.aggregate_parent)

    def __call__(self, run_order: Optional[List[Union[List[str], str]]] = None):
        self.calculator(run_order)

    def save(self, configs_path: PurePath):
        orders = {k: i for i, k in enumerate(self.calculator.topological_order)}
        configs = {}
        for k, v in self.parser.types.items():
            configs[k] = {
                'name': k,
                'type': str(v),
                'value': Value.to_yaml(self.parser.values.get(k, None)),
                'order': orders.get(k, -1),
                'options': list(self.parser.options.get(k, {}).keys()),
                'computed': Value.to_yaml(getattr(self.calculator.configs, k, None)),
                'is_hyperparam': self.parser.hyperparams.get(k, None),
                'is_explicitly_specified': (k in self.parser.explicitly_specified)
            }

        with open(str(configs_path), "w") as file:
            file.write(util.yaml_dump(configs))

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
                    value = Value.to_str(value)

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
            value_str = Value.to_str(value)

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


def load_configs(configs_path: PurePath):
    configs_path = Path(configs_path)
    if not configs_path.exists():
        return None

    with open(str(configs_path), 'r') as file:
        configs = util.yaml_load(file.read())

    return configs
