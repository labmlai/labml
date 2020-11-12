from pathlib import PurePath, Path
from typing import Dict

from labml.internal import util
from .base import Configs
from .utils import Value

_CONFIG_PRINT_LEN = 40


class ConfigProcessor:
    def __init__(self, configs: Configs, values: Dict[str, any] = None):
        self.values = values
        self.configs = configs

    def to_json(self):
        return self.configs.to_json()

    def save(self, configs_path: PurePath):
        with open(str(configs_path), "w") as file:
            file.write(util.yaml_dump(self.to_json()))

    def get_hyperparams(self):
        hyperparams = {}
        for k, v in self.to_json().items():
            if v['is_hyperparam'] or v['is_explicitly_specified']:
                value = v['value'] or v['computed']
                if type(value) not in {int, float, str}:
                    value = Value.to_str(value)

                hyperparams[k] = value

        return hyperparams


def load_configs(configs_path: Path):
    if not configs_path.exists():
        return None

    with open(str(configs_path), 'r') as file:
        configs = util.yaml_load(file.read())

    return configs
