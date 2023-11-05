from pathlib import Path
from typing import Optional, Set

from labml import logger
from labml.logger import Text
from . import CONFIGS_FOLDER
from .. import util
from ..app.configs import AppTrackConfigs
from ..lab import get_app_url_for_handle


class Computer:
    app_sync_url: str
    app_polling_url: str
    app_configs: AppTrackConfigs
    uuid: str
    config_folder: Path

    def __init__(self):
        self.home = Path.home()
        self.config_folder = self.home / CONFIGS_FOLDER
        self.projects_folder = self.config_folder / 'projects'
        self.runs_cache = self.config_folder / 'runs_cache'
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
                ('~/.labml/configs.yaml', Text.value),
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

        app_url = get_app_url_for_handle('', base_url=config['app_url'])

        if not app_url:
            raise ValueError('Please provide `app_url` in `~/labml/configs.yaml` or set '
                             'environment variable `labml_app_url`.')

        self.app_configs = AppTrackConfigs(url=get_app_url_for_handle('computer', base_url=app_url),
                                           frequency=config['app_track_frequency'],
                                           open_browser=config['app_open_browser'],
                                           is_default=False)
        self.app_sync_url = get_app_url_for_handle('sync', base_url=app_url)
        self.app_polling_url = get_app_url_for_handle('polling', base_url=app_url)

    def __str__(self):
        return f"<Computer uuid={self.uuid}>"

    def __repr__(self):
        return str(self)

    @staticmethod
    def __default_config():
        return dict(
            app_url=None,
            app_track_frequency=0,
            app_open_browser=True,
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
