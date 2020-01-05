import os
import sqlite3
from typing import List, Union
from typing import Optional

import numpy as np
import tensorflow as tf
from matplotlib.axes import Axes
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from lab import util
from lab.experiment import experiment_run
from lab.lab import Lab


def get_lab():
    return Lab(os.getcwd())


def get_run_info(experiment_name: str, run_index: Optional[int] = None):
    lab = get_lab()
    experiment_path = lab.experiments / experiment_name
    if run_index is None:
        run_index = experiment_run.get_last_run_index(experiment_path, None, False)
    run_path = experiment_path / str(run_index)
    run_info_path = run_path / 'run.yaml'

    with open(str(run_info_path), 'r') as f:
        data = util.yaml_load(f.read())
        run = experiment_run.RunInfo.from_dict(experiment_path, data)

    return run


class Dir:
    def __init__(self, options):
        self.__options = {k.replace('.', '_'): v for k, v in options.items()}
        self.__list = [k for k in self.__options.keys()]

    def __dir__(self):
        return self.__list

    def __getattr__(self, k):
        return self.__options[k]


class TensorBoardAnalyzer:
    def __init__(self, log_path):
        self.event_acc = EventAccumulator(str(log_path))

    def load(self):
        self.event_acc.Reload()

    def tensor(self, name):
        name = name.replace('.', '/')
        return self.event_acc.Tensors(name)

    @staticmethod
    def summarize(events):
        step = np.mean([e.step for e in events])
        values = np.sort([tf.make_ndarray(e.tensor_proto) for e in events])
        basis_points = np.percentile(values, [
            0,
            6.68,
            15.87,
            30.85,
            50.00,
            69.15,
            84.13,
            93.32,
            100.00
        ])

        return np.concatenate(([step], basis_points))

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
        if len(merged) > 0:
            results.append(self.summarize(merged))

        return np.array(results)

    @staticmethod
    def summarize_compressed_histogram(events):
        basis_points = [
            0,
            668,
            1587,
            3085,
            5000,
            6915,
            8413,
            9332,
            10000
        ]
        results = []
        for e in events:
            assert (len(e.compressed_histogram_values) == len(basis_points))
            for i, c in enumerate(e.compressed_histogram_values):
                assert (c.basis_point == basis_points[i])
            results.append([e.step] + [c.value for c in e.compressed_histogram_values])

        return np.asarray(results)

    @staticmethod
    def render_density(ax: Axes, data, color, name, *,
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

    def render_scalar(self, name, ax: Axes, color, *, levels=5, line_width=1, alpha=0.6):
        data = self.summarize_scalars(self.tensor(name))
        self.render_density(ax, data, color, name,
                            levels=levels,
                            line_width=line_width,
                            alpha=alpha)

    def render_histogram(self, name, ax: Axes, color, *, levels=5, line_width=1, alpha=0.6):
        data = self.summarize_compressed_histogram(self.histogram(name))
        self.render_density(ax, data, color, name,
                            levels=levels,
                            line_width=line_width,
                            alpha=alpha)

    @staticmethod
    def render_matrix(matrix, ax, color):
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection

        rows, cols = matrix.shape
        x_ticks = [matrix[0, i] for i in range(1, cols)]
        y_ticks = [matrix[i, 0] for i in range(1, rows)]

        max_density = 0
        for y in range(rows - 2):
            for x in range(cols - 2):
                area = (x_ticks[x + 1] - x_ticks[x]) * (y_ticks[y + 1] - y_ticks[y])
                density = matrix[y + 1, x + 1] / area
                if max_density < density:
                    max_density = density

        boxes = []
        for y in range(rows - 2):
            for x in range(cols - 2):
                width = x_ticks[x + 1] - x_ticks[x]
                height = y_ticks[y + 1] - y_ticks[y]
                area = width * height
                density = matrix[y + 1, x + 1] / area
                density = max(density, 1)
                # alpha = np.log(density) / np.log(max_density)
                alpha = density / max_density
                rect = Rectangle((x_ticks[x], y_ticks[y]), width, height,
                                 alpha=alpha, color=color)
                boxes.append(rect)

        pc = PatchCollection(boxes, match_original=True)

        ax.add_collection(pc)

        # Plot something
        _ = ax.errorbar([], [], xerr=[], yerr=[],
                        fmt='None', ecolor='k')

        return x_ticks[0], x_ticks[-1], y_ticks[0], y_ticks[-1]

    def render_tensors(self, tensors: Union[str, List[any]], axes: np.ndarray, color):
        if type(tensors) == str:
            tensors = self.tensor(tensors)
        assert len(axes.shape) == 2
        assert axes.shape[0] * axes.shape[1] == len(tensors)
        x_min = x_max = y_min = y_max = 0

        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                idx = i * axes.shape[1] + j
                axes[i, j].set_title(f"{tensors[idx].step :,}")
                matrix = tf.make_ndarray(tensors[idx].tensor_proto)
                x1, x2, y1, y2 = self.render_matrix(matrix,
                                                    axes[i, j],
                                                    color)
                x_min = min(x_min, x1)
                x_max = max(x_max, x2)
                y_min = min(y_min, y1)
                y_max = max(y_max, y2)

        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                axes[i, j].set_xlim(x_min, x_max)
                axes[i, j].set_ylim(y_min, y_max)


class SQLiteAnalyzer:
    def __init__(self, sqlite_path):
        self.conn = sqlite3.connect(str(sqlite_path))

    def get_key(self, name):
        """
        ## Get a tensor summary

        If 'name' is 'None' it returns a list of all available tensors.
        """
        return name

    def scalar(self, name):
        key = self.get_key(name)
        cur = self.conn.execute(
            f'SELECT step, value from scalars WHERE indicator = '
            f'"{key}"')
        return [c for c in cur]

    @staticmethod
    def summarize(events):
        step = np.mean([e[0] for e in events])
        values = np.sort([e[1] for e in events])
        basis_points = np.percentile(values, [
            0,
            6.68,
            15.87,
            30.85,
            50.00,
            69.15,
            84.13,
            93.32,
            100.00
        ])

        return np.concatenate(([step], basis_points))

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
        if len(merged) > 0:
            results.append(self.summarize(merged))

        return np.array(results)

    @staticmethod
    def summarize_compressed_histogram(events):
        basis_points = [
            0,
            668,
            1587,
            3085,
            5000,
            6915,
            8413,
            9332,
            10000
        ]
        results = []
        for e in events:
            assert (len(e.compressed_histogram_values) == len(basis_points))
            for i, c in enumerate(e.compressed_histogram_values):
                assert (c.basis_point == basis_points[i])
            results.append([e.step] + [c.value for c in e.compressed_histogram_values])

        return np.asarray(results)

    @staticmethod
    def render_density(ax: Axes, data, color, name, *,
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

    def render_scalar(self, name, ax: Axes, color, *, levels=5, line_width=1, alpha=0.6):
        data = self.summarize_scalars(self.scalar(name))
        self.render_density(ax, data, color, name,
                            levels=levels,
                            line_width=line_width,
                            alpha=alpha)

    def render_histogram(self, name, ax: Axes, color, *, levels=5, line_width=1, alpha=0.6):
        data = self.summarize_compressed_histogram(self.histogram(name))
        self.render_density(ax, data, color, name,
                            levels=levels,
                            line_width=line_width,
                            alpha=alpha)

    @staticmethod
    def render_matrix(matrix, ax, color):
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection

        rows, cols = matrix.shape
        x_ticks = [matrix[0, i] for i in range(1, cols)]
        y_ticks = [matrix[i, 0] for i in range(1, rows)]

        max_density = 0
        for y in range(rows - 2):
            for x in range(cols - 2):
                area = (x_ticks[x + 1] - x_ticks[x]) * (y_ticks[y + 1] - y_ticks[y])
                density = matrix[y + 1, x + 1] / area
                if max_density < density:
                    max_density = density

        boxes = []
        for y in range(rows - 2):
            for x in range(cols - 2):
                width = x_ticks[x + 1] - x_ticks[x]
                height = y_ticks[y + 1] - y_ticks[y]
                area = width * height
                density = matrix[y + 1, x + 1] / area
                density = max(density, 1)
                # alpha = np.log(density) / np.log(max_density)
                alpha = density / max_density
                rect = Rectangle((x_ticks[x], y_ticks[y]), width, height,
                                 alpha=alpha, color=color)
                boxes.append(rect)

        pc = PatchCollection(boxes, match_original=True)

        ax.add_collection(pc)

        # Plot something
        _ = ax.errorbar([], [], xerr=[], yerr=[],
                        fmt='None', ecolor='k')

        return x_ticks[0], x_ticks[-1], y_ticks[0], y_ticks[-1]

    def render_tensors(self, tensors: Union[str, List[any]], axes: np.ndarray, color):
        if type(tensors) == str:
            tensors = self.tensor(tensors)
        assert len(axes.shape) == 2
        assert axes.shape[0] * axes.shape[1] == len(tensors)
        x_min = x_max = y_min = y_max = 0

        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                idx = i * axes.shape[1] + j
                axes[i, j].set_title(f"{tensors[idx].step :,}")
                matrix = tf.make_ndarray(tensors[idx].tensor_proto)
                x1, x2, y1, y2 = self.render_matrix(matrix,
                                                    axes[i, j],
                                                    color)
                x_min = min(x_min, x1)
                x_max = max(x_max, x2)
                y_min = min(y_min, y1)
                y_max = max(y_max, y2)

        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                axes[i, j].set_xlim(x_min, x_max)
                axes[i, j].set_ylim(y_min, y_max)


class Analyzer:
    def __init__(self, experiment_name: str, run_index: Optional[int] = None):
        self.run_info = get_run_info(experiment_name)
        self.tb = TensorBoardAnalyzer(self.run_info.tensorboard_log_path)
        self.sqlite = SQLiteAnalyzer(self.run_info.sqlite_path)

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

        return [Dir(dirs[k]) for k in args]
#
#
# if __name__ == '__main__':
#     print(get_run_info('mnist_loop'))
