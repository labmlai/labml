from pathlib import PurePath
from typing import Dict, Optional, List, Union

from labml import logger
from labml.internal import util
from labml.internal.configs.base import Configs
from labml.internal.configs.utils import Value
from labml.logger import Text

from .calculator import Calculator
from .parser import Parser

_CONFIG_PRINT_LEN = 40


class ConfigProcessorDict:
    def __init__(self, configs: Dict[str, any], values: Dict[str, any] = None):
        self.configs = configs
        if values is None:
            values = {}
        self.values = values

    def __call__(self, run_order: Optional[List[Union[List[str], str]]] = None):
        self.configs.update(self.values)

    def to_json(self):
        orders = {k: i for i, k in enumerate(self.configs.keys())}
        configs = {}
        for k, v in self.configs.items():
            configs[k] = {
                'name': k,
                'type': str(type(v)),
                'value': Value.to_yaml(v),
                'order': orders.get(k, -1),
                'options': [],
                'computed': Value.to_yaml(v),
                'is_hyperparam': False,
                'is_meta': False,
                'is_explicitly_specified': (k in self.values)
            }

        return configs

    def save(self, configs_path: PurePath):
        configs = self.to_json()

        with open(str(configs_path), "w") as file:
            file.write(util.yaml_dump(configs))

    def get_hyperparams(self):
        return self.values.copy()

    def __print_config(self, key, *, value=None):
        parts = ['\t']

        is_hyperparam = key in self.values

        if is_hyperparam:
            parts.append((key, [Text.key, Text.highlight]))
        else:
            parts.append((key, Text.key))

        parts.append((' = ', Text.subtle))

        value_str = Value.to_str(value)

        value_str = value_str.replace('\n', '')
        if len(value_str) < _CONFIG_PRINT_LEN:
            parts.append((f"{value_str}", Text.value))
        else:
            parts.append((f"{value_str[:_CONFIG_PRINT_LEN]}...", Text.value))
        parts.append('\t')

        return parts

    def print(self):
        order = list(self.configs.keys())
        order.sort()

        logger.log("Configs:", Text.heading)

        for k in order:
            parts = self.__print_config(k, value=self.configs[k])

            logger.log(parts)

        logger.log()
