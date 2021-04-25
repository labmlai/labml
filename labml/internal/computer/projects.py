from pathlib import Path

import labml.internal.lab


class Project:
    def __init__(self, project_path: Path):
        self.lab = labml.internal.lab.Lab(project_path)

    def get_runs(self):
        from labml.internal.manage.runs import get_runs
        runs = list(get_runs(self.lab.experiments))

        print(runs)


def _test():
    from labml.internal.computer.configs import computer_singleton
    projects = [Project(Path(p)) for p in computer_singleton().get_projects()]
    for p in projects:
        p.get_runs()


if __name__ == '__main__':
    _test()
