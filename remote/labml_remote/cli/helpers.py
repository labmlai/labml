from typing import List, Tuple

import click

import labml_remote.distributed.torch_launch
from . import util


@click.command()
@click.option('--python-cmd', '--cmd', required=True, type=click.STRING, help='Python command to run')
@click.option('--nproc-per-node', required=True, type=click.INT, help='Number of processes per node')
@click.option('--use-env', is_flag=True,
              help='Whether not to pass local_rank as a command line argument.')
@click.option('--master-port', default=1234, type=click.INT, help='Master port')
@click.option('--env', type=(str, str), multiple=True, help='Environment variables')
@click.option('--tag', type=click.STRING, multiple=True, help='Tags for the job')
def helper_torch_launch(python_cmd: str, nproc_per_node: int, use_env: bool,
                        master_port: int, env: List[Tuple[str, str]],
                        tag: List[str]):
    """A replacement for torch.distributed.launch."""
    if not tag:
        tag = [t.strip() for t in python_cmd.split(' ') if t.strip()]
        tag += ['distributed']
        tag += ['torch']
    labml_remote.distributed.torch_launch.launch(python_cmd,
                                                 n_proc_per_node=nproc_per_node,
                                                 use_env=use_env,
                                                 master_port=master_port,
                                                 env_vars=util.get_env_dict(env),
                                                 tags=tag)
