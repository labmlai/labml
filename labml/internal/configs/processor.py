from pathlib import Path
from typing import Dict, Union, Optional, List

from labml.internal import util
from .base import Configs
from .utils import Value

_CONFIG_PRINT_LEN = 40


class ConfigsSaver:
    def save(self, configs: Dict):
        raise NotImplementedError


class FileConfigsSaver(ConfigsSaver):
    def __init__(self, configs_path: Path):
        self.configs_path = configs_path

    def save(self, configs: Dict):
        with open(str(self.configs_path), "w") as file:
            file.write(util.yaml_dump(configs))


class ConfigProcessor:
    configs_dict: Optional[Dict[str, any]]
    configs: Optional[Configs]
    savers: List[ConfigsSaver]

    def __init__(self, configs: Union[Configs, Dict[str, any]], values: Dict[str, any] = None):
        if values is None:
            values = {}

        if isinstance(configs, Configs):
            self.configs_dict = None
            self.configs = configs
            self.configs._set_update_callback(self._on_configs_updated)
            self.configs._set_values(values)
        elif isinstance(configs, dict):
            self.configs = None
            self.configs_dict = configs
            self.configs_dict.update(values)

        self.values = values
        self.savers = []

    def _on_configs_updated(self):
        configs = self.to_json()
        for s in self.savers:
            s.save(configs)

    def to_json(self):
        if self.configs is not None:
            return self.configs._to_json()
        else:
            orders = {k: i for i, k in enumerate(self.configs_dict.keys())}
            configs = {}
            for k, v in self.configs_dict.items():
                configs[k] = {
                    'name': k,
                    'type': str(type(v)),
                    'value': Value.to_yaml_truncated(v),
                    'order': orders.get(k, -1),
                    'options': [],
                    'computed': Value.to_yaml_truncated(v),
                    'is_hyperparam': False,
                    'is_meta': False,
                    'is_explicitly_specified': (k in self.values)
                }

            return configs

    def add_saver(self, saver: ConfigsSaver):
        self.savers.append(saver)
        saver.save(self.to_json())

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
