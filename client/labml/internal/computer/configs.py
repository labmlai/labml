from pathlib import Path
from typing import Optional, Set, Union

from labml import logger, monit
from labml.internal import util
from labml.internal.api.configs import WebAPIConfigs
from labml.internal.computer import CONFIGS_FOLDER
from labml.logger import Text


class Computer:
    """
    ### Computer

    Lab contains the labml specific properties.
    """
    web_api_sync: str
    web_api_polling: str
    web_api: WebAPIConfigs
    uuid: str
    config_folder: Path

    def __init__(self):
        self.home = Path.home()
        self.config_folder = self.home / CONFIGS_FOLDER
        self.projects_folder = self.config_folder / 'projects'
        self.runs_cache = self.config_folder / 'runs_cache'
        self.tensorboard_symlink_dir = self.config_folder / 'tensorboard'
        self.configs_file = self.config_folder / 'configs.yaml'
        self.app_folder = self.config_folder / 'app'

        self.__load_configs()

    def __load_configs(self):
        if self.config_folder.is_file():
            self.config_folder.unlink()

        if not self.config_folder.exists():
            self.config_folder.mkdir(parents=True)

        if not self.projects_folder.exists():
            self.projects_folder.mkdir()

        if not self.app_folder.exists():
            self.app_folder.mkdir()

        if not self.runs_cache.exists():
            self.runs_cache.mkdir()

        if self.configs_file.exists():
            with open(str(self.configs_file)) as f:
                config = util.yaml_load(f.read())
                if config is None:
                    config = {}
        else:
            logger.log([
                ('~/labml/configs.yaml', Text.value),
                ' does not exist. Creating ',
                (str(self.configs_file), Text.meta)])
            config = {}

        if 'uuid' not in config:
            from uuid import uuid1
            config['uuid'] = uuid1().hex
            with open(str(self.configs_file), 'w') as f:
                f.write(util.yaml_dump(config))

        default_config = self.__default_config()
        for k, v in default_config.items():
            if k not in config:
                config[k] = v

        self.uuid = config['uuid']
        web_api_url = config['web_api']
        if web_api_url[0:4] != 'http':
            web_api_url = f"https://api.labml.ai/api/v1/computer?labml_token={web_api_url}&"
        self.web_api = WebAPIConfigs(url=web_api_url,
                                     frequency=config['web_api_frequency'],
                                     verify_connection=config['web_api_verify_connection'],
                                     open_browser=config['web_api_open_browser'],
                                     is_default=web_api_url == self.__default_config()['web_api'])
        self.web_api_sync = config['web_api_sync']
        self.web_api_polling = config['web_api_polling']

        self.tensorboard_port = config['tensorboard_port']
        self.tensorboard_visible_port = config['tensorboard_visible_port']
        self.tensorboard_host = config['tensorboard_host']
        self.tensorboard_protocol = config['tensorboard_protocol']

    def set_token(self, token: str):
        with monit.section('Update ~/labml/configs.yaml'):
            with open(str(self.configs_file)) as f:
                config = util.yaml_load(f.read())
                assert config is not None

            config['web_api'] = token

            with open(str(self.configs_file), 'w') as f:
                f.write(util.yaml_dump(config))

    def __str__(self):
        return f"<Computer uuid={self.uuid}>"

    def __repr__(self):
        return str(self)

    @staticmethod
    def __default_config():
        return dict(
            web_api='https://api.labml.ai/api/v1/computer?',
            web_api_frequency=0,
            web_api_verify_connection=True,
            web_api_open_browser=True,
            web_api_sync='https://api.labml.ai/api/v1/sync?',
            web_api_polling='https://api.labml.ai/api/v1/polling?',
            tensorboard_port=6006,
            tensorboard_visible_port=6006,
            tensorboard_host='localhost',
            tensorboard_protocol='http',
        )

    def get_projects(self) -> Set[str]:
        projects = set()
        to_remove = []

        for p in self.projects_folder.iterdir():
            with open(str(p), 'r') as f:
                project_path = f.read()
            if project_path in projects:
                to_remove.append(p)
            else:
                if Path(project_path).exists():
                    projects.add(project_path)
                else:
                    to_remove.append(p)

        for p in to_remove:
            p.unlink()

        return projects

    def add_project(self, path: Path):
        project_path = str(path.absolute())
        if project_path in self.get_projects():
            return

        from uuid import uuid1
        p_uuid = uuid1().hex

        with open(str(self.projects_folder / f'{p_uuid}.txt'), 'w') as f:
            f.write(project_path)


_internal: Optional[Computer] = None


def computer_singleton() -> Computer:
    global _internal
    if _internal is None:
        _internal = Computer()

    return _internal
