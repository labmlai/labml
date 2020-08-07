from pathlib import PurePath, Path
from typing import Dict, Optional, List, Union

from labml import logger
from labml.internal import util
from labml.logger import Text
from .base import Configs
from .calculator import Calculator
from .parser import Parser
from .utils import Value

_CONFIG_PRINT_LEN = 40


class ConfigProcessor:
    def __init__(self, configs: Configs, values: Dict[str, any] = None, *,
                 is_directly_specified: bool = True):
        self.parser = Parser(configs, values, is_directly_specified=is_directly_specified)
        self.calculator = Calculator(configs=configs,
                                     options=self.parser.options,
                                     evals=self.parser.evals,
                                     types=self.parser.types,
                                     values=self.parser.values,
                                     secondary_values=self.parser.secondary_values,
                                     aggregate_parent=self.parser.aggregate_parent)

    def __call__(self, run_order: Optional[List[Union[List[str], str]]] = None):
        self.calculator(run_order)

    def save(self, configs_path: PurePath):
        configs = self.to_json()

        with open(str(configs_path), "w") as file:
            file.write(util.yaml_dump(configs))

    def _get_options_list(self, key: str):
        opts = list(self.parser.options.get(key, {}).keys())
        if not opts:
            opts = list((self.parser.aggregates.get(key, {}).keys()))

        return opts

    def to_json(self):
        orders = {k: i for i, k in enumerate(self.calculator.topological_order)}
        configs = {}
        for k, v in self.parser.types.items():
            configs[k] = {
                'name': k,
                'type': str(v),
                'value': Value.to_yaml(self.parser.values.get(k, None)),
                'order': orders.get(k, -1),
                'options': self._get_options_list(k),
                'computed': Value.to_yaml(getattr(self.calculator.configs, k, None)),
                'is_hyperparam': self.parser.hyperparams.get(k, None),
                'is_meta': self.parser.meta.get(k, None),
                'is_explicitly_specified': (k in self.parser.explicitly_specified)
            }

        for k, proc in self.calculator.config_processors.items():
            sub_configs = proc.to_json()
            for sk, v in sub_configs.items():
                configs[f"{k}.{sk}"] = v

        return configs

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

    def __print_config(self, key, *, indentation: int, value=None, option=None,
                       other_options=None, is_ignored=False, is_list=False):
        logger.log(self.__config_log_parts(key,
                                           indentation=indentation,
                                           value=value,
                                           option=option,
                                           other_options=other_options,
                                           is_ignored=is_ignored,
                                           is_list=is_list))
        if key in self.calculator.config_processors:
            self.calculator.config_processors[key].print_configs(indentation + 1)

    def __config_log_parts(self, key, *, indentation: int, value=None, option=None,
                           other_options=None, is_ignored=False, is_list=False):
        parts = ['\t' * indentation]

        if is_ignored:
            parts.append((key, Text.subtle))
            return parts

        is_hyperparam = self.parser.hyperparams.get(key, None)
        is_meta = self.parser.meta.get(key, None)
        if is_hyperparam is None:
            is_hyperparam = key in self.parser.explicitly_specified
            if is_meta:
                is_hyperparam = False

        if is_hyperparam:
            parts.append((key, [Text.key, Text.highlight]))
        elif is_meta:
            parts.append((key, Text.subtle))
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

    def print_configs(self, indentation: int):
        order = self.calculator.topological_order.copy()
        order.sort()
        added = set(order)
        ignored = set()

        for k in self.parser.types:
            if k not in added:
                added.add(k)
                order.append(k)
                ignored.add(k)

        for k in order:
            computed = getattr(self.calculator.configs, k, None)
            opts = self._get_options_list(k)
            if k in ignored:
                self.__print_config(k, indentation=indentation, is_ignored=True)
            elif opts:
                v = self.parser.values[k]
                if v in opts:
                    opts.remove(v)
                else:
                    v = None

                self.__print_config(k,
                                    indentation=indentation,
                                    value=computed,
                                    option=v,
                                    other_options=opts)
            else:
                self.__print_config(k, indentation=indentation, value=computed)

    def print(self):
        logger.log("Configs:", Text.heading)
        self.print_configs(1)
        logger.log()


def load_configs(configs_path: PurePath):
    configs_path = Path(configs_path)
    if not configs_path.exists():
        return None

    with open(str(configs_path), 'r') as file:
        configs = util.yaml_load(file.read())

    return configs
