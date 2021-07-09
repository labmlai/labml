from typing import Union, List, Tuple, Optional

import click

from labml.logger import Text
from labml_remote.configs import Configs
from labml_remote.job import JOBS, Job
from labml_remote.server import SERVERS

SERVER_CHOICES = list(iter(SERVERS))

SERVER_OPTION_ATTRS = dict(multiple=True,
                           type=click.Choice(SERVER_CHOICES),
                           help='Server name(s). Blank for all servers')

if len(SERVER_CHOICES) == 1:
    SINGLE_SERVER_OPTION_ATTRS = dict(type=click.Choice(SERVER_CHOICES),
                                      default=SERVER_CHOICES[0],
                                      help='Server name.')
else:
    SINGLE_SERVER_OPTION_ATTRS = dict(type=click.Choice(SERVER_CHOICES),
                                      prompt=True,
                                      help='Server name.')


def get_servers(server: Union[str, List[str], None]) -> List[str]:
    if not server:
        return list(Configs.get().servers.keys())
    elif type(server) is str:
        return [server]
    else:
        return server


def get_env_dict(env: List[Tuple[str, str]]):
    return {k: v for k, v in env}


def log_job(_job: Job):
    log_parts = [(_job.job_key, Text.meta)]
    log_parts += [': ', (_job.job_id, Text.subtle)]
    log_parts += [' [', (_job.server.conf.name, Text.key), '] ']
    if _job.running:
        log_parts += [' (', (str(_job.pid), Text.meta2), ') ']
    log_parts += [' {']
    for i, t in enumerate(_job.tags):
        if i != 0:
            log_parts += [',']
        log_parts += [(t, Text.meta)]
    log_parts += ['} ']

    log_parts += [': ', (_job.command, Text.value)]

    return log_parts


def get_jobs(job: Union[str, List[str], None], tags: List[str], *,
             exclude_tags: Optional[List[str]] = None,
             is_master: bool = True,
             is_running: bool = True) -> List[str]:
    if isinstance(job, str):
        return [job]
    elif job:
        return job

    if not tags:
        click.echo("No tags provided")

    if exclude_tags is None:
        exclude_tags = ['__hidden__']
    jobs = JOBS.filter_by_tags(tags)
    jobs = JOBS.filter_out_by_tags(exclude_tags, jobs)
    if is_master:
        master_jobs = JOBS.filter_by_tags(['master'], jobs)
        if 0 < len(master_jobs) < len(jobs):
            click.echo(f"Selecting jobs tagged 'master' {len(master_jobs)}/{len(jobs)}")
            jobs = master_jobs
    if is_running:
        running_jobs = JOBS.filter_running(jobs)
        if 0 < len(running_jobs) < len(jobs):
            click.echo(f"Selecting running jobs {len(running_jobs)}/{len(jobs)}")
            jobs = running_jobs

    return [j.job_key for j in jobs]
