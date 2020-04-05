import os
import sqlite3

import numpy as np
from matplotlib.axes import Axes

from lab import util
from lab.experiment import experiment_run
from lab.lab import Lab


def get_lab():
    return Lab(os.getcwd())


def get_run_info(experiment_name: str, run_uuid: str):
    lab = get_lab()
    experiment_path = lab.experiments / experiment_name

    run_path = experiment_path / run_uuid
    run_info_path = run_path / 'run.yaml'

    with open(str(run_info_path), 'r') as f:
        data = util.yaml_load(f.read())
        run = experiment_run.RunInfo.from_dict(experiment_path, data)

    return run


class Dir:
    def __init__(self, options):
        self.__options = {k.replace('.', '_'): v for k, v in options.items()}
        self.__list = [k for k in self.__options]

    def __dir__(self):
        return self.__list

    def __getattr__(self, k):
        return self.__options[k]


_BASIS_POINTS = [
    0,
    6.68,
    15.87,
    30.85,
    50.00,
    69.15,
    84.13,
    93.32,
    100.00
]


class Analytics:
    def render_density(self, ax: Axes, data, color, name, *,
                       levels=5,
                       line_width=1,
                       alpha=0.6):
        # Mean line
        ln = ax.plot(data[:, 0], data[:, 5],
                     lw=line_width,
                     color=color,
                     alpha=1,
                     label=name)

        # Other percentiles
        for i in range(1, levels):
            ax.fill_between(
                data[:, 0],
                data[:, 5 - i],
                data[:, 5 + i],
                color=color,
                lw=0,
                alpha=alpha ** i)

        return ln

    def summarize(self, events):
        raise NotImplementedError()

    def summarize_scalars(self, events):
        # Shrink to 100 histograms
        interval = max(1, len(events) // 100)

        merged = []
        results = []
        for i, e in enumerate(events):
            if i > 0 and (i + 1) % interval == 0:
                results.append(self.summarize(merged))
                merged = []
            merged.append(e)
        if merged:
            results.append(self.summarize(merged))

        return np.array(results)


class SQLiteAnalytics(Analytics):
    def __init__(self, sqlite_path):
        self.conn = sqlite3.connect(str(sqlite_path))

    def get_key(self, name):
        return name

    def scalar(self, name):
        key = self.get_key(name)
        cur = self.conn.execute(
            f'SELECT step, value from scalars WHERE indicator = '
            f'"{key}"')
        return [c for c in cur]

    def summarize(self, events):
        step = np.mean([e[0] for e in events])
        values = np.sort([e[1] for e in events])
        basis_points = np.percentile(values, _BASIS_POINTS)

        return np.concatenate(([step], basis_points))

    def render_scalar(self, name, ax: Axes, color, *, levels=5, line_width=1, alpha=0.6):
        data = self.summarize_scalars(self.scalar(name))
        self.render_density(ax, data, color, name,
                            levels=levels,
                            line_width=line_width,
                            alpha=alpha)


class Analyzer:
    def __init__(self, experiment_name: str, run_uuid: str):
        self.run_info = get_run_info(experiment_name, run_uuid)
        self.sqlite = SQLiteAnalytics(self.run_info.sqlite_path)

        with open(str(self.run_info.indicators_path), 'r') as f:
            self.indicators = util.yaml_load(f.read())

    def get_indicators(self, *args):
        # TODO: Need to handle Queue's and mean scalars of histograms

        dirs = {k: {} for k in args}

        def add(class_name, key, value):
            if class_name not in dirs:
                return
            dirs[class_name][key] = value

        for k, v in self.indicators.items():
            cn = v['class_name']
            add(cn, k, k)
            if cn == 'Histogram':
                add('Scalar', k, f"{k}.mean")
            if cn == 'Queue':
                add('Scalar', k, f"{k}.mean")
                add('Histogram', k, k)
            if cn == 'IndexedScalar':
                add('Scalar', k, k)

        return [Dir(dirs[k]) for k in args]


def _test():
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_context('notebook')
    sns.set_style('white')
    a = Analyzer('mnist_loop', '6d10ba48730c11ea8223acde48001122')
    Histograms, Scalars = a.get_indicators('Histogram', 'Scalar')
    _, ax = plt.subplots(figsize=(18, 9))

    a.sqlite.render_scalar(Scalars.train_loss, ax, sns.color_palette()[0])
    plt.show()


if __name__ == '__main__':
    _test()
