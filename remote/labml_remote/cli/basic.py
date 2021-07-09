from pathlib import Path
from typing import List, Tuple

import click

from labml_remote.execute import UIMode
from labml_remote.server import SERVERS
import labml_remote.configs.defaults
from . import util
from ..job import JOBS

_SHOW_OUTPUT_ATTRS = dict(is_flag=True, help='Show full output of the command.')


@click.command()
def init():
    """Initialize the project to use labml_remote. You should add servers to .remote/configs.yaml."""
    labml_remote.configs.defaults.create_default_project(Path('.'))

@click.command()
@click.option('--server', **util.SERVER_OPTION_ATTRS)
@click.option('--show-output', **_SHOW_OUTPUT_ATTRS)
def setup(server: str, show_output: bool):
    """Install Python on the server with conda"""
    for k in util.get_servers(server):
        SERVERS[k].setup(ui_mode=UIMode.full if show_output else UIMode.dots)
    JOBS.start_status_watcher()


@click.command()
@click.option('--server', **util.SERVER_OPTION_ATTRS)
@click.option('--show-output', **_SHOW_OUTPUT_ATTRS)
def rsync(server: str, show_output: bool):
    """RSync the contents of this project with the server"""
    for k in util.get_servers(server):
        SERVERS[k].rsync(ui_mode=UIMode.full if show_output else UIMode.dots)


@click.command()
@click.option('--server', **util.SERVER_OPTION_ATTRS)
@click.option('--show-output', **_SHOW_OUTPUT_ATTRS)
def update_packages(server: str, show_output: bool):
    """Update Python packages on the server"""
    for k in util.get_servers(server):
        SERVERS[k].update_packages(ui_mode=UIMode.full if show_output else UIMode.dots)


@click.command()
@click.option('--server', **util.SERVER_OPTION_ATTRS)
@click.option('--show-output', **_SHOW_OUTPUT_ATTRS)
def prepare(server: str, show_output: bool):
    """Setup python, Rsync and Update pip packages"""
    for k in util.get_servers(server):
        SERVERS[k].setup(ui_mode=UIMode.full if show_output else UIMode.dots)
        SERVERS[k].rsync(ui_mode=UIMode.full if show_output else UIMode.dots)
        SERVERS[k].update_packages(ui_mode=UIMode.full if show_output else UIMode.dots)
    JOBS.start_status_watcher()


@click.command()
@click.option('--server', **util.SINGLE_SERVER_OPTION_ATTRS)
@click.option('--cmd', required=True, type=click.STRING, help='Command to execute.')
@click.option('--env', type=(str, str), multiple=True, help='Environment variables.')
@click.option('--silent', is_flag=True, help='Hide the output.')
def run(server: str, cmd: str, silent: bool, env: List[Tuple[str, str]]):
    """Run  a command on a given server in the Python environment"""
    return SERVERS[server].command(cmd, util.get_env_dict(env),
                                   ui_mode=UIMode.dots if silent else UIMode.full,
                                   is_background=False,
                                   is_eval=False)
