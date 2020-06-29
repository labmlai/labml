from pathlib import PurePath, Path
from typing import List, Optional

from labml.internal import util
from labml.utils import get_caller_file

_CONFIG_FILE_NAME = '.labml.yaml'


class LabYamlNotfoundError(RuntimeError):
    pass


class Lab:
    """
    ### Lab

    Lab contains the labml specific properties.
    """

    def __init__(self):
        self.indicators = {}
        self.path = None
        self.check_repo_dirty = None
        self.data_path = None
        self.experiments = None
        self.web_api = None

        python_file = get_caller_file()
        self.set_path(python_file)

    def set_path(self, path: str):
        configs = self.__get_config_files(path)

        if len(configs) == 0:
            raise LabYamlNotfoundError(f"No '.labml.yaml' config file found."
                                       f"Looking in {path}")

        config = self.__get_config(configs)

        self.path = PurePath(config['path'])
        self.check_repo_dirty = config['check_repo_dirty']
        self.data_path = self.path / config['data_path']
        self.experiments = self.path / config['experiments_path']
        self.indicators = config['indicators']
        self.web_api = config['web_api']

    def __str__(self):
        return f"<Lab path={self.path}>"

    def __repr__(self):
        return str(self)

    @staticmethod
    def __get_config(configs):
        config = dict(
            path=None,
            check_repo_dirty=False,
            is_log_python_file=True,
            config_file_path=None,
            data_path='data',
            experiments_path='logs',
            analytics_path='analytics',
            analytics_templates={},
            web_api=None,
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

        for i, c in enumerate(reversed(configs)):
            if config['path'] is None:
                config['path'] = c['config_file_path']

            assert 'path' not in c
            assert i == 0 or 'experiments_path' not in c
            assert i == 0 or 'analytics_path' not in c

            for k, v in c.items():
                if k not in config:
                    raise RuntimeError(f"Unknown config parameter #{k} in file "
                                       f"{c['config_file_path'] / _CONFIG_FILE_NAME}")
                else:
                    config[k] = v

        return config

    @staticmethod
    def __get_config_files(path: str):
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
