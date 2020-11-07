from labml import lab
from labml.logger import inspect


def get_runs():
    runs = {}
    for exp_path in lab.get_experiments_path().iterdir():
        if exp_path.name.startswith('_'):
            continue
        for run_path in exp_path.iterdir():
            runs[run_path.name] = run_path

    return runs


if __name__ == '__main__':
    inspect(get_runs())
