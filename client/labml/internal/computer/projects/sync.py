from typing import Dict, Optional, List

from labml.internal.computer.projects import Projects
from labml.internal.computer.projects.run_summary import RunSummary
from labml.internal.manage.runs import remove_run


class SyncRuns:
    runs: Dict[str, RunSummary]
    projects: Optional[Projects]

    def __init__(self):
        from labml.internal.computer.projects.api import DirectApiCaller
        from labml.internal.computer.configs import computer_singleton

        self.sync_caller = DirectApiCaller(computer_singleton().web_api_sync,
                                           {'computer_uuid': computer_singleton().uuid},
                                           timeout_seconds=15)

        self.projects = None
        self.runs = {}

    def load(self):
        self.projects = Projects()
        self.runs = self.projects.get_runs()

    def get_runs(self, uuids: List[str]):
        missing = False

        for r in uuids:
            if r not in self.runs:
                missing = True

        if missing:
            self.load()

        runs = []
        for r in uuids:
            if r in self.runs:
                runs.append(self.runs[r])

        return runs

    def sync(self):
        self.load()

        from labml import logger
        from labml.logger import Text

        response = self.sync_caller.send({'runs': [r.to_dict() for r in self.runs.values()]})

        status = response['runs']
        # for k, v in status.items():
        #     logger.log(k, Text.title)
        #     inspect(v)

        for r in status['deleted']:
            remove_run(self.runs[r].path)
            self.runs[r].clear_cache()

        if status['deleted']:
            logger.log('Removed runs:')
            for r in status['deleted']:
                logger.log([r, ': ', (str(self.runs[r].path), Text.subtle)])
        if status['unknown']:
            logger.log('Not synced:', Text.warning)
            for r in status['unknown']:
                logger.log([r, ': ', (str(self.runs[r].path), Text.subtle)])


def _main():
    s = SyncRuns()
    s.sync()


if __name__ == '__main__':
    _main()
