from pathlib import Path
from typing import Dict, List

import labml.internal.lab
from labml.internal.computer.projects.run_summary import RunSummary


class Project:
    runs: Dict[str, RunSummary]

    def __init__(self, project_path: Path):
        self.lab = labml.internal.lab.Lab(project_path)
        self.load_runs()

    def load_runs(self):
        from labml.internal.manage.runs import get_runs
        run_paths = list(get_runs(self.lab.experiments))
        runs = [RunSummary(p) for p in run_paths]
        self.runs = {r.uuid: r for r in runs}


class Projects:
    projects: List[Project]

    def __init__(self):
        self.load()

    def get_runs(self):
        return [r for p in self.projects for r in p.runs.values()]

    def load(self):
        from labml.internal.computer.configs import computer_singleton
        self.projects = [Project(Path(p)) for p in computer_singleton().get_projects()]

    def _get_run(self, uuid):
        for p in self.projects:
            if uuid in p.runs:
                return p.runs[uuid]
        return None

    def get_or_load_run(self, uuid):
        if self._get_run(uuid) is None:
            self.load()

        return self._get_run(uuid)


def _test():
    projects = Projects()
    runs = projects.get_runs()

    from labml.internal.computer.projects.api import DirectApiCaller
    from labml.internal.computer.configs import computer_singleton

    sync_caller = DirectApiCaller(computer_singleton().web_api_sync,
                                  {'computer_uuid': computer_singleton().uuid},
                                  timeout_seconds=15)

    sync_caller.send([r.to_dict() for r in runs])


if __name__ == '__main__':
    _test()
