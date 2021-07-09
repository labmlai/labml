from pathlib import Path
from typing import Dict, List

from labml.logger import Text

from labml import logger

from labml.internal.experiment.experiment_run import RunLoadError

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
        runs = []
        for p in run_paths:
            try:
                run = RunSummary(p)
                runs.append(run)
            except RunLoadError as e:
                logger.log([('Error loading run: ', Text.warning), (str(p), Text.value)])
                print(e)

        self.runs = {r.uuid: r for r in runs}


class Projects:
    projects: List[Project]

    def __init__(self):
        self.load()

    def get_runs(self):
        return {r.uuid: r for p in self.projects for r in p.runs.values()}

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
    from labml.logger import inspect

    projects = Projects()
    runs = projects.get_runs()
    for r in runs:
        inspect(r.to_dict())


if __name__ == '__main__':
    _test()
