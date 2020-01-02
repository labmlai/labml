from typing import Optional

from lab import util
from lab.experiment import experiment_run
from lab.lab import Lab


def get_lab():
    return Lab(__file__)


def get_run_info(experiment_name: str, run_index: Optional[int] = None):
    lab = get_lab()
    experiment_path = lab.experiments / experiment_name
    if run_index is None:
        run_index = experiment_run.get_last_run_index(experiment_path, None, False)
    run_path = experiment_path / str(run_index)
    run_info_path = run_path / 'run.yaml'

    with open(str(run_info_path), 'r') as f:
        data = util.yaml_load(f.read())
        run = experiment_run.Run.from_dict(experiment_path, data)

    print(run)


if __name__ == '__main__':
    get_run_info('mnist_loop')