import warnings
from pathlib import PurePath, Path
from typing import List, Optional, Dict

from labml.internal import util
from labml.internal.util import is_colab, is_kaggle
from labml.utils import get_caller_file

_CONFIG_FILE_NAME = '.labml.yaml'


class LabYamlNotfoundError(RuntimeError):
    pass


class WebAPIConfigs:
    url: str
    frequency: float
    verify_connection: bool

    def __init__(self, *,
                 url: str,
                 frequency: float,
                 verify_connection: bool):
        self.frequency = frequency
        self.verify_connection = verify_connection
        self.url = url


class Lab:
    """
    ### Lab

    Lab contains the labml specific properties.
    """
    web_api: Optional[WebAPIConfigs]

    def __init__(self):
        self.indicators = {}
        self.path = None
        self.check_repo_dirty = None
        self.data_path = None
        self.experiments = None
        self.web_api = None
        self.configs = self.__default_config()
        self.custom_configs = []
        self.__update_configs()

        python_file = get_caller_file()
        self.__load_configs(python_file)

    def set_path(self, path: str):
        self.__load_configs(path)

    def __load_configs(self, path: str):
        config_files = self.__load_config_files(path)

        if not config_files:
            if not is_colab() and not is_kaggle():
                warnings.warn(f"No '.labml.yaml' config file found. "
                              f"Looking in {path}",
                              UserWarning, stacklevel=4)

        for c in config_files:
            self.__merge_configs(c)

        for c in self.custom_configs:
            self.__merge_configs(c)

        if not config_files and self.configs['path'] is None:
            self.configs['path'] = path

        self.__update_configs()

    def __update_configs(self):
        if self.configs['path'] is None:
            self.path = None
            self.experiments = None
            self.data_path = None
        else:
            self.path = PurePath(self.configs['path'])
            self.data_path = self.path / self.configs['data_path']
            self.experiments = self.path / self.configs['experiments_path']

        self.check_repo_dirty = self.configs['check_repo_dirty']
        self.indicators = self.configs['indicators']
        self.web_api = WebAPIConfigs(url=self.configs['web_api'],
                                     frequency=self.configs['web_api_frequency'],
                                     verify_connection=self.configs['web_api_verify_connection'])

    def set_configurations(self, configs: Dict[str, any]):
        self.custom_configs.append(configs)
        for c in self.custom_configs:
            self.__merge_configs(c)
        self.__update_configs()

    def __str__(self):
        return f"<Lab path={self.path}>"

    def __repr__(self):
        return str(self)

    @staticmethod
    def __default_config():
        return dict(
            path=None,
            check_repo_dirty=False,
            is_log_python_file=True,
            config_file_path=None,
            data_path='data',
            experiments_path='logs',
            analytics_path='analytics',
            analytics_templates={},
            web_api=None,
            web_api_frequency=60,
            web_api_verify_connection=True,
            indicators=[{
                'class_name': 'Scalar',
                'is_print': True,
                'name': '*'
            }, {
                'class_name': 'Scalar',
                'is_print': False,
                'name': 'param.*'
            }, {
                'class_name': 'Scalar',
                'is_print': False,
                'name': 'module.*'
            }
            ]
        )

    def __merge_configs(self, c):
        if self.configs['path'] is None and 'config_file_path' in c:
            self.configs['path'] = c['config_file_path']

        for k, v in c.items():
            if k not in self.configs:
                raise RuntimeError(f"Unknown config parameter #{k} in file "
                                   f"{c['config_file_path'] / _CONFIG_FILE_NAME}")
            elif k == 'indicators':
                self.configs[k] += v
            else:
                self.configs[k] = v

    @staticmethod
    def __load_config_files(path: str):
        path = Path(path).resolve()
        configs = []

        while path.exists():
            if path.is_dir():
                config_file = path / _CONFIG_FILE_NAME
                if config_file.is_file():
                    with open(str(config_file)) as f:
                        config = util.yaml_load(f.read())
                        if config is None:
                            config = {}
                        config['config_file_path'] = path
                        configs.append(config)

            if str(path) == path.root:
                break

            path = path.parent

        return configs

    def get_experiments(self) -> List[Path]:
        """
        Get list of experiments
        """
        experiments_path = Path(self.experiments)
        return [child for child in experiments_path.iterdir()]


_internal: Optional[Lab] = None


def lab_singleton() -> Lab:
    global _internal
    if _internal is None:
        _internal = Lab()

    return _internal
