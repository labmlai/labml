from matplotlib.axes import Axes
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import numpy as np

from lab.experiment import ExperimentInfo
from lab.lab import Lab


class Analyzer:
    """
    # TensorBoard Summary Analyzer

    This loads TensorBoard summaries, and provides a set of tools to
    create customized charts on Jupyter notebooks.

    The data format we use is as follows 👇

    ```
    [index,
     0%,
     6.68%,
     15.87%,
     30.85%,
     50.00%,
     69.15%,
     84.13%,
     93.32%,
     100.00%]
    ```

    Each data point gives a histogram in the the above format.
    """

    def __init__(self, lab: Lab, experiment: str):
        self.info = ExperimentInfo(lab, experiment)
        self.event_acc = EventAccumulator(str(self.info.summary_path))

    def load(self):
        """
        ## Load summaries
        """
        self.event_acc.Reload()

    def scalar(self, name=None):
        """
        ## Get a scalar summary

        If 'name' is 'None' it returns a list of all available scalars.
        """
        if name is None:
            return self.event_acc.Tags()['scalars']

        return self.event_acc.Scalars(name)

    def histogram(self, name=None):
        """
        ## Get a histogram summary

        If 'name' is 'None' it returns a list of all available histograms.
        """
        if name is None:
            return self.event_acc.Tags()['histograms']

        return self.event_acc.CompressedHistograms(name)

    def summarize(self, events):
        """
        ## Merge many data points and get a distribution
        """

        step = np.mean([e.step for e in events])
        values = np.sort([e.value for e in events])
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

    def summarize_series(self, events):
        """
        ### Shrink data points and produce a histogram
        """

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

    def summarize_compressed_histogram(self, events):
        """
        ## Convert a TensorBoard histogram to our format
        """
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

    def render_density(self, ax: Axes, data, color, name, *,
                       levels=5,
                       line_width=1,
                       alpha=0.6):
        """
        ## Render a density plot from data
        """

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
        """
        ## Summarize and render a scalar
        """
        data = self.summarize_series(self.scalar(name))
        self.render_density(ax, data, color, name,
                            levels=levels,
                            line_width=line_width,
                            alpha=alpha)

    def render_histogram(self, name, ax: Axes, color, *, levels=5, line_width=1, alpha=0.6):
        """
        ## Summarize and render a histogram
        """
        data = self.summarize_compressed_histogram(self.histogram(name))
        self.render_density(ax, data, color, name,
                            levels=levels,
                            line_width=line_width,
                            alpha=alpha)
