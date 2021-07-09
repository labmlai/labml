"""
eval "$(_LABML_REMOTE_COMPLETE=source_zsh labml_remote)"
"""

import click

from labml_remote.cli import util, job, helpers, basic
from labml_remote.server import SERVERS


@click.group()
def main():
    if len(list(SERVERS)) == 0:
        click.echo('No servers found. Run labml_remote init to initialize a project and edit '
                   '.remote/configs.yaml and add servers.')


main: click.Group
main.add_command(basic.init)
main.add_command(basic.setup)
main.add_command(basic.rsync)
main.add_command(basic.update_packages)
main.add_command(basic.prepare)
main.add_command(basic.run)

main.add_command(job.job_run)
main.add_command(job.job_rsync)
main.add_command(job.job_list)
main.add_command(job.job_tail)
main.add_command(job.job_kill)

main.add_command(helpers.helper_torch_launch)

if __name__ == '__main__':
    main()
