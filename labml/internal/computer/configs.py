from pathlib import Path
from typing import Optional

from labml import logger, monit
from labml.internal import util
from labml.internal.computer import CONFIGS_FOLDER
from labml.internal.lab import WebAPIConfigs
from labml.logger import Text


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
        self.config_folder = self.home / CONFIGS_FOLDER
        self.configs_file = self.config_folder / 'configs.yaml'

        self.__load_configs()

    def __load_configs(self):
        if self.config_folder.is_file():
            self.config_folder.unlink()

        if not self.config_folder.exists():
            self.config_folder.mkdir(parents=True)


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
        )


_internal: Optional[Computer] = None


def computer_singleton() -> Computer:
    global _internal
    if _internal is None:
        _internal = Computer()

    return _internal
