from pathlib import Path

from labml.internal.computer.projects.run_summary import RunSummary

import labml.internal.lab


class Project:
    def __init__(self, project_path: Path):
        self.lab = labml.internal.lab.Lab(project_path)

    def get_runs(self):
        from labml.internal.manage.runs import get_runs
        run_paths = list(get_runs(self.lab.experiments))
        runs = [RunSummary(p) for p in run_paths]
        print(runs)


def _test():
    from labml.internal.computer.configs import computer_singleton
    projects = [Project(Path(p)) for p in computer_singleton().get_projects()]
    for p in projects:
        p.get_runs()


if __name__ == '__main__':
    _test()
