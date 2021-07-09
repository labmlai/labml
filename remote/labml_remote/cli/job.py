import time
from typing import List, Tuple

import click

from labml import logger, monit
from labml.logger import Text
from labml_remote.execute import UIMode
from labml_remote.job import JOBS
from labml_remote.server import SERVERS
from . import util


@click.command()
@click.option('--server', **util.SINGLE_SERVER_OPTION_ATTRS)
@click.option('--cmd', required=True, type=click.STRING, help='Command to run as a job')
@click.option('--env', type=(str, str), multiple=True, help='Environment variables')
@click.option('--tag', multiple=True, type=click.STRING,
              help='Tags for the job. Defaults to "custom" and "no-tags" if not provided')
def job_run(server: str, cmd: str, env: List[Tuple[str, str]], tag: List[str]):
    """Start a job"""
    if not tag:
        tag = ['custom', 'no-tags']
    JOBS.create(server, cmd, util.get_env_dict(env), tag).start()


@click.command()
@click.option('--server', **util.SERVER_OPTION_ATTRS)
@click.option('--delay', type=click.INT, default=5, help="Refresh delay. 0 not to watch")
@click.option('--show-output', is_flag=True, help='Show output of rsync')
def job_rsync(server: str, delay: int, show_output: bool):
    """RSync job outputs from server"""
    for k in util.get_servers(server):
        SERVERS[k].rsync_jobs(ui_mode=UIMode.full if show_output else UIMode.dots)
    if delay > 0:
        while True:
            logger.log('Watching...', Text.meta)
            time.sleep(delay)
            for k in util.get_servers(server):
                SERVERS[k].rsync_jobs()


@click.command()
@click.option('--rsync', 'is_rsync_before', is_flag=True, help='Whether to RSync and then list')
@click.option('--stopped', 'show_stopped', is_flag=True, help='Show finished/stopped jobs')
@click.option('--hidden', 'show_hidden', is_flag=True, help='Show hidden jobs')
@click.option('--tag', multiple=True, type=click.STRING, help='Filter by tags')
def job_list(is_rsync_before: bool, show_stopped: bool, show_hidden: bool, tag: List[str]):
    """Show list of jobs"""
    if is_rsync_before:
        for k in util.get_servers(''):
            SERVERS[k].rsync_jobs()
        time.sleep(0.5)

    for _job in JOBS.all():
        matched = True
        for t in tag:
            if t not in _job.tags:
                matched = False
        if not matched:
            continue
        if not show_hidden and '__hidden__' in _job.tags:
            continue
        if not show_stopped and not _job.running:
            continue

        logger.log(util.log_job(_job))


@click.command()
@click.option('--job', type=click.Choice(list(JOBS.job_keys())), help='Job to tail')
@click.option('--tag', multiple=True, type=click.STRING, help='Find job to tail by tags')
@click.option('--delay', type=click.INT, default=5, help="Refresh delay. 0 not to watch")
def job_tail(job: str, tag: List[str], delay: int):
    """Tail job output"""
    jobs = util.get_jobs(job, tag)
    if len(jobs) > 1:
        click.echo(f"Selecting a job out of {len(jobs)}")
    job = jobs[0]

    _job = JOBS.by_key(job)
    logger.log(util.log_job(_job))
    _job.tail()
    if delay <= 0:
        return
    while not _job.stopped:
        with monit.section('rsync', is_silent=True):
            _job.server.rsync_jobs(ui_mode=UIMode.none, is_silent=True)
        time.sleep(0.5)
        _job.update_stopped()
        _job.tail()
        time.sleep(delay)


@click.command()
@click.option('--job', multiple=True, type=click.Choice(list(JOBS.job_keys())), help='Jobs to kill')
@click.option('--tag', multiple=True, type=click.STRING, help='Find jobs to kill by tag')
@click.option('--signal', type=click.STRING, default='SIGKILL', help='Kill signal')
def job_kill(job: List[str], tag: List[str], signal: str):
    """Kill jobs"""
    if not job and not tag:
        if not click.confirm("Killing all jobs. Do you want to continue?"):
            return 1

    jobs = util.get_jobs(job, tag, is_master=False)

    for j in jobs:
        _job = JOBS.by_key(j)
        if not _job.running:
            continue
        log_parts = util.log_job(_job)
        res = _job.server.shell(f'kill -{signal} {_job.pid}')
        if res.exit_code == 0:
            log_parts += [(' KILLED', Text.success)]
            logger.log(log_parts)
        else:
            log_parts += [(' FAILED', Text.success)]
            logger.log(log_parts)
            logger.log(res.out)
            logger.log(res.err, Text.warning)
