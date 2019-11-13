from pathlib import PurePath, Path
from typing import List

from lab import util

_CONFIG_FILE_NAME = '.lab.yaml'


class Lab:
    """
    ### Lab

    Lab contains the lab specific properties.
    """

    def __init__(self, path: str):
        configs = self.__get_config_files(path)

        if len(configs) == 0:
            raise RuntimeError("No '.lab.yaml' config file found.")

        config = self.__get_config(configs)

        self.path = PurePath(config['path'])
        self.check_repo_dirty = config['check_repo_dirty']
        self.is_log_python_file = config['is_log_python_file']

    @staticmethod
    def __get_config(configs):
        config = dict(
            path=None,
            check_repo_dirty=True,
            is_log_python_file=True,
            config_file_path=None
        )

        for c in reversed(configs):
            if 'path' in c:
                c['path'] = c['config_file_path'] / c['path']
            elif config['path'] is None:
                c['path'] = c['config_file_path']

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

    @property
    def experiments(self) -> PurePath:
        """
        ### Experiments path
        """
        return self.path / "logs"

    def get_experiments(self) -> List[Path]:
        """
        Get list of experiments
        """
        experiments_path = Path(self.experiments)
        return [child for child in experiments_path.iterdir()]
