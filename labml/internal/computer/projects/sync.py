from typing import Dict

from labml.internal.computer.projects import Projects
from labml.internal.computer.projects.run_summary import RunSummary
from labml.internal.manage.runs import remove_run


class SyncRuns:
    def __init__(self):
        from labml.internal.computer.projects.api import DirectApiCaller
        from labml.internal.computer.configs import computer_singleton

        self.sync_caller = DirectApiCaller(computer_singleton().web_api_sync,
                                           {'computer_uuid': computer_singleton().uuid},
                                           timeout_seconds=15)

    def sync(self):
        projects = Projects()
        runs = projects.get_runs()

        from labml import logger
        from labml.logger import inspect, Text

        response = self.sync_caller.send({'runs': [r.to_dict() for r in runs]})

        status = response['runs']
        # for k, v in status.items():
        #     logger.log(k, Text.title)
        #     inspect(v)

        runs_dict: Dict[str, RunSummary] = {r.uuid: r for r in runs}
        for r in status['deleted']:
            remove_run(runs_dict[r].path)
            runs_dict[r].clear_cache()

        if status['deleted']:
            logger.log('Removed runs:')
            for r in status['deleted']:
                logger.log([r, ': ', (str(runs_dict[r].path), Text.subtle)])
        if status['unknown']:
            logger.log('Not synced:', Text.warning)
            for r in status['unknown']:
                logger.log([r, ': ', (str(runs_dict[r].path), Text.subtle)])


def _main():
    s = SyncRuns()
    s.sync()


if __name__ == '__main__':
    _main()
