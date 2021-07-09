import os
import stat
import sys
from pathlib import Path
from typing import Dict, Optional

import paramiko
from paramiko import SSHClient
from scp import SCPClient

from labml import monit, logger
from labml_remote import util
from labml_remote.configs import Configs
from labml_remote.errors import RemoteError
from labml_remote.execute import UIMode
from labml_remote.execute.local import LocalExecutor
from labml_remote.execute.remote import RemoteExecutor, EvalResult
from labml_remote.util import get_env_vars


class Server:
    def __init__(self, server_id: str):
        conf = Configs.get()
        self.project_name = conf.name
        self.scripts_folder = conf.template_scripts_folder
        self.conf = conf.servers[server_id]
        self.__client = None
        self.__home_path = None
        self.__remote_executor = None
        self.__local_executor = None

    @property
    def client(self) -> SSHClient:
        if self.__client is None:
            self.__client = SSHClient()
            self.__client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            with monit.section(f'Connecting to {self.conf.hostname}'):
                self.__client.connect(hostname=self.conf.hostname,
                                      username=self.conf.username,
                                      pkey=self.conf.private_key,
                                      password=self.conf.password)
        return self.__client

    @property
    def home_path(self) -> str:
        if self.__home_path is None:
            self.__home_path = self.remote_exec.eval('pwd', log_dir=None).out

        return self.__home_path

    @property
    def remote_exec(self) -> RemoteExecutor:
        if self.__remote_executor is None:
            self.__remote_executor = RemoteExecutor(self.client)

        return self.__remote_executor

    @property
    def local_exec(self) -> LocalExecutor:
        if self.__local_executor is None:
            self.__local_executor = LocalExecutor()

        return self.__local_executor

    @property
    def remote_scripts_path(self):
        return f'{self.home_path}/{self.project_name}/{Configs.get().remote_scripts_folder_name}'

    def setup(self, *, ui_mode: UIMode = UIMode.dots):
        _ = self.client
        with monit.section(f"Setup server {self.conf.name}"):
            logger.log()
            if self.remote_exec.eval(f'test -d {self.project_name}', log_dir=None).exit_code != 0:
                self.remote_exec.eval(f'mkdir {self.project_name}', log_dir=None)
            if self.remote_exec.eval(f'test -d {self.remote_scripts_path}', log_dir=None).exit_code != 0:
                self.remote_exec.eval(f'mkdir {self.remote_scripts_path}', log_dir=None)

            python_version = f'{sys.version_info.major}.{sys.version_info.minor}'

            script = self.template_script('setup.sh', {
                'python_version': python_version,
            })

            if self.copy_and_run_script(script, 'setup.sh',
                                        ui_mode=ui_mode).exit_code != 0:
                raise RemoteError("Failed to setup server")

    def copy_script(self, script: str, script_name: str):
        scripts_path = Configs.get().project_scripts_folder

        if not scripts_path.exists():
            scripts_path.mkdir()

        script_file = scripts_path / script_name
        with open(str(script_file), 'w') as f:
            f.write(script)

        os.chmod(str(script_file), stat.S_IRWXU | stat.S_IRWXG)
        scp = SCPClient(self.client.get_transport())
        remote_path = f'{self.remote_scripts_path}/{script_name}'
        scp.put(str(script_file), remote_path)

        return remote_path

    def run_script(self, remote_path: str, *,
                   log_dir: Optional[Path],
                   ui_mode: UIMode = UIMode.dots,
                   is_background=False,
                   is_eval=False) -> EvalResult:
        if is_eval:
            return self.remote_exec.eval(remote_path,
                                         log_dir=log_dir)
        elif is_background:
            return self.remote_exec.background(remote_path,
                                               log_dir=log_dir, ui_mode=ui_mode)
        else:
            return self.remote_exec.stream(remote_path,
                                           log_dir=log_dir, ui_mode=ui_mode)

    @staticmethod
    def _get_log_folder(name: str):
        n = 1
        while True:
            folder_name = f'{name}_{n :04d}'
            log_dir = Configs.get().project_logs_folder / folder_name
            if not log_dir.exists():
                log_dir.mkdir(parents=True)
                return log_dir
            n += 1

    def copy_and_run_script(self, script: str, script_name: str, *,
                            ui_mode: UIMode = UIMode.dots,
                            is_background=False,
                            is_eval=False) -> EvalResult:
        remote_path = self.copy_script(script, script_name)
        logger_name = script_name.replace('.', '__')
        log_dir = self._get_log_folder(f'{logger_name}_{self.conf.name}')

        return self.run_script(remote_path, log_dir=log_dir,
                               ui_mode=ui_mode,
                               is_background=is_background,
                               is_eval=is_eval)

    def rsync(self, *, ui_mode: UIMode = UIMode.dots):
        with monit.section(f"RSync {self.conf.name}"):
            logger.log()
            exclude_path = Configs.get().exclude_file
            exclude_path = exclude_path.absolute()
            # z = compress
            # r = recursive
            # a = equivalent to (-rlptgoD) archive (recursive/preserve everything)
            # v = verbose
            # u = update (skip whats newer on receiver)
            # K = keep symlinks
            # L = transform links to dir
            # t = preserve modification times
            # l = copy links
            # p = preserve permissions
            # g = preserve group
            # o = preserve owner
            # D = preserve device files
            rsync_cmd = ['rsync', '-zravuKLt', '--executability']
            if self.conf.private_key_file is not None:
                rsync_cmd += ['-e', f'"ssh -o StrictHostKeyChecking=no -i {self.conf.private_key_file}"']
            else:
                rsync_cmd += ['-e', f'"ssh -o StrictHostKeyChecking=no"']
            if exclude_path.exists():
                rsync_cmd += [f"--exclude-from='{str(exclude_path)}'"]
            rsync_cmd += ['./']  # source
            rsync_cmd += [f'{self.conf.username}@{self.conf.hostname}:~/{self.project_name}/']  # destination

            log_dir = self._get_log_folder(f'rsync_{self.conf.name}')
            exit_code = self.local_exec.stream(' '.join(rsync_cmd),
                                               log_dir=log_dir,
                                               ui_mode=ui_mode)

            if exit_code != 0:
                raise RemoteError("Failed to run rsync")

    def rsync_jobs(self, *, ui_mode: UIMode = UIMode.dots, is_silent=False):
        with monit.section(f"RSync {self.conf.name} jobs", is_silent=is_silent):
            if not is_silent:
                logger.log()
            rsync_cmd = ['rsync', '-zravuKLt', '--executability']
            if self.conf.private_key_file is not None:
                rsync_cmd += ['-e', f'"ssh -o StrictHostKeyChecking=no -i {self.conf.private_key_file}"']
            else:
                rsync_cmd += ['-e', f'"ssh -o StrictHostKeyChecking=no"']
            rsync_cmd += [f'{self.conf.username}@{self.conf.hostname}:'
                          f'~/{self.project_name}/{Configs.get().remote_jobs_folder_name}/']
            rsync_cmd += [str(Configs.get().project_jobs_folder)]

            log_dir = self._get_log_folder(f'rsync_jobs_{self.conf.name}')
            exit_code = self.local_exec.stream(' '.join(rsync_cmd),
                                               log_dir=log_dir,
                                               ui_mode=ui_mode)

            if exit_code != 0:
                raise RemoteError("Failed to run rsync")

    def update_packages(self, *, ui_mode: UIMode = UIMode.dots):
        _ = self.client
        with monit.section(f"Update python packages {self.conf.name}"):
            logger.log()

            pipfile = Path('.') / 'Pipfile'
            requirements = Path('.') / 'requirements.txt'

            script = self.template_script('update.sh', {
                'has_pipfile': str(pipfile.exists()),
                'has_requirements': str(requirements.exists())
            })

            if self.copy_and_run_script(script, 'update.sh',
                                        ui_mode=ui_mode).exit_code != 0:
                raise RemoteError("Failed to update packages")

    def shell(self, command: str):
        return self.remote_exec.eval(command, log_dir=None)

    def command(self, command: str, env_vars: Dict[str, str], *,
                ui_mode: UIMode = UIMode.dots,
                is_background: bool, is_eval: bool):
        _ = self.client
        with monit.section("Run command"):
            logger.log()
            pipfile = Path('.') / 'Pipfile'
            # requirements = Path('.') / 'requirements.txt'

            script = self.template_script('run.sh', {
                'use_pipenv': str(pipfile.exists()),
                'run_command': command,
                'environment_variables': get_env_vars(env_vars)
            })

            res = self.copy_and_run_script(script, 'run.sh',
                                           ui_mode=ui_mode,
                                           is_background=is_background, is_eval=is_eval)

            if res.exit_code != 0:
                raise RemoteError("Failed to run command")

            return res

    def template_script(self, script_name: str, replace: Dict[str, str]):
        replace['name'] = self.project_name
        replace['home'] = self.home_path

        return util.template(self.scripts_folder / script_name, replace)

    def script(self, script_name: str, replace: Dict[str, str], *,
               ui_mode: UIMode = UIMode.dots,
               is_background: bool, is_eval: bool) -> EvalResult:
        _ = self.client
        with monit.section("Run script"):
            logger.log()

            script = self.template_script(script_name, replace)

            res = self.copy_and_run_script(script, 'script.sh',
                                           ui_mode=ui_mode,
                                           is_background=is_background, is_eval=is_eval)

            if res.exit_code != 0:
                raise RemoteError("Failed to run command")

            return res


class ServerCollection:
    def __init__(self):
        self._servers = {}
        self.load_all()

    def __iter__(self):
        return iter(self._servers.keys())

    def __getitem__(self, server_id: str) -> Server:
        server_id = str(server_id)
        if server_id not in self._servers:
            self._servers[server_id] = Server(server_id)

        return self._servers[server_id]

    def load_all(self):
        for s in Configs.get().servers:
            if s not in self._servers:
                self._servers[s] = Server(s)


SERVERS = ServerCollection()
