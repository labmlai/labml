import os
import stat
from pathlib import Path
from typing import Optional, Dict, Any, Union

import paramiko
import yaml

_CONFIGS = None

_DOT_REMOTE_PATH = Path('.') / '.remote'
_CONFIGS_FILE_NAME = 'configs.yaml'


class ServerConfig:
    name: str
    private_key_file: Optional[str]
    username: Optional[str]
    hostname: Optional[str]
    password: Optional[str]
    private_key: Optional[paramiko.RSAKey]
    properties: Dict[str, any]

    def __init__(self, name: str, configs: Dict[str, any]):
        self.name = name
        self.hostname = configs.pop('hostname')
        self.username = configs.pop('username', 'ubuntu')
        self.password = configs.pop('password', None)
        self.private_key_file = configs.pop('private_key', None)
        if self.private_key_file is not None:
            os.chmod(str(self.private_key_file), stat.S_IRUSR | stat.S_IWUSR)
            self.private_key = paramiko.RSAKey.from_private_key_file(self.private_key_file)
        else:
            self.private_key = None

        self.properties = configs


class Configs:
    name: str
    template_scripts_folder: Path
    servers: Dict[str, ServerConfig]
    project_scripts_folder: Path
    project_logs_folder: Path
    project_jobs_folder: Path
    exclude_file: Path
    remote_scripts_folder_name: str
    remote_jobs_folder_name: str

    def __init__(self, configs: Dict[str, any]):
        self.servers = {str(k): ServerConfig(k, s) for k, s in configs.get('servers', {}).items()}
        self.name = configs.get('name', Path('..').parent.absolute().name)
        self.project_scripts_folder = Path('.') / configs.get('scripts_folder', '.remote/scripts')
        self.project_logs_folder = Path('.') / configs.get('logs_folder', '.remote/logs')
        self.project_jobs_folder = Path('.') / configs.get('jobs_folder', '.remote/jobs')
        self.exclude_file = Path('.') / configs.get('exclude_file', '.remote/exclude.txt')
        self.remote_scripts_folder_name = configs.get('remote_scripts_folder_name', '.remote-scripts')
        self.remote_jobs_folder_name = configs.get('remote_jobs_folder_name', '.jobs')

        self.template_scripts_folder = Path(__file__).absolute().parent.parent / 'scripts'

    @staticmethod
    def _create(path: Path = _DOT_REMOTE_PATH / _CONFIGS_FILE_NAME):
        if not path.exists():
            return Configs({})
        with open(str(path), 'r') as f:
            return Configs(yaml.load(f.read(), yaml.FullLoader))

    @staticmethod
    def get() -> 'Configs':
        global _CONFIGS
        if _CONFIGS is None:
            _CONFIGS = Configs._create()
        return _CONFIGS
