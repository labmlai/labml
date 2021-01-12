from pathlib import Path
from typing import Optional

from labml import monit
from labml.internal import util
from labml.internal.computer import CONFIGS_FOLDER
from labml.internal.lab import WebAPIConfigs


class Computer:
    """
    ### Computer

    Lab contains the labml specific properties.
    """
    web_api: WebAPIConfigs
    uuid: str
    config_folder: Path

    def __init__(self):
        self.home = Path.home()

        self.__load_configs()

    def __load_configs(self):
        self.config_folder = self.home / CONFIGS_FOLDER

        print(self.config_folder, self.config_folder.exists())
        if self.config_folder.is_file():
            self.config_folder.unlink()

        if not self.config_folder.exists():
            self.config_folder.mkdir(parents=True)

        configs_file = self.config_folder / 'configs.yaml'

        if configs_file.exists():
            with open(str(configs_file)) as f:
                config = util.yaml_load(f.read())
                if config is None:
                    config = {}
        else:
            with monit.section('Creating a .labml/config.yaml'):
                from uuid import uuid1
                config = {'uuid': uuid1().hex}
                with open(str(configs_file), 'w') as f:
                    f.write(util.yaml_dump(config))

        default_config = self.__default_config()
        for k, v in default_config.items():
            if k not in config:
                config[k] = v

        self.uuid = config['uuid']
        web_api_url = config['web_api']
        if web_api_url[0:4] != 'http':
            web_api_url = f"https://api.lab-ml.com/api/v1/computer?labml_token={web_api_url}&"
        self.web_api = WebAPIConfigs(url=web_api_url,
                                     frequency=config['web_api_frequency'],
                                     verify_connection=config['web_api_verify_connection'],
                                     open_browser=config['web_api_open_browser'])

    def __str__(self):
        return f"<Computer uuid={self.uuid}>"

    def __repr__(self):
        return str(self)

    @staticmethod
    def __default_config():
        return dict(
            web_api='https://api.lab-ml.com/api/v1/computer?',
            web_api_frequency=0,
            web_api_verify_connection=True,
            web_api_open_browser=True,
        )


_internal: Optional[Computer] = None


def computer_singleton() -> Computer:
    global _internal
    if _internal is None:
        _internal = Computer()

    return _internal
