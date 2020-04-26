from typing import List

from . import AltairAnalytics
from ..tensorboard import TensorBoardAnalytics


class AltairTensorBoardAnalytics(TensorBoardAnalytics, AltairAnalytics):
    r"""
    This will be created by :class:`lab.analytics.Analyzer`
    """

    def render_scalar(self, name, *,
                      line_color='steelblue', range_color='steelblue',
                      levels: int = 5, alpha: float = 0.6,
                      points: int = 100,
                      height: int = 400, width: int = 800, height_minimap: int = 100):
        r"""
        Creates a compressed line chart with distribution

        Arguments:
            name(str): name of the scalar to visualize.
                The names are usually taken from objects returned by
                :meth:`lab.analytics.Analyzer.get_indicators`.

            line_color: color of the line
            range_color: color of the distribution if the line chart is compressed
            levels: how many levels of the distribution to be plotted
            alpha: opacity of the distribution
            points: number of points to plot
            height: height of the visualization
            width: width of the visualization
            height_minimap: height of the view finder

        Return:
            The Altair visualization

        Example:
            >>> a.tb.render_scalar(Scalars.train_loss)
        """
        data = self.summarize_scalars(self.tensor(name), points=points)
        return self.render_density_minimap(data,
                                           name=name,
                                           line_color=line_color,
                                           range_color=range_color,
                                           levels=levels,
                                           alpha=alpha,
                                           width=width,
                                           height=height,
                                           height_minimap=height_minimap)

    def render_histogram(self, name, *,
                         line_color='steelblue', range_color='steelblue',
                         levels: int = 5, alpha: int = 0.6,
                         height: int = 400, width: int = 800, height_minimap: int = 100):
        data = self.summarize_compressed_histogram(self.tensor(name))
        return self.render_density_minimap(data,
                                           name=name,
                                           line_color=line_color,
                                           range_color=range_color,
                                           levels=levels,
                                           alpha=alpha,
                                           width=width,
                                           height=height,
                                           height_minimap=height_minimap)

    def render_histogram_multiple(self, names: List[str], *,
                                  levels: int = 5, alpha: int = 0.6,
                                  height: int = 400, width: int = 800, height_minimap: int = 100):
        datas = [self.summarize_compressed_histogram(self.tensor(n)) for n in names]
        return self.render_density_minimap_multiple(datas,
                                                    names=names,
                                                    levels=levels,
                                                    alpha=alpha,
                                                    width=width,
                                                    height=height,
                                                    height_minimap=height_minimap)

    def render_scalar_simple(self, name, *,
                             line_color='steelblue', range_color='steelblue',
                             levels: int = 5, alpha: float = 0.6,
                             points: int = 100,
                             height: int = 400, width: int = 800):
        data = self.summarize_scalars(self.tensor(name), points=points)
        return self.render_density(data,
                                   name=name,
                                   line_color=line_color,
                                   range_color=range_color,
                                   levels=levels,
                                   alpha=alpha,
                                   width=width,
                                   height=height)

    def render_histogram_simple(self, name, *,
                                line_color='steelblue', range_color='steelblue',
                                levels: int = 5, alpha: int = 0.6,
                                height: int = 400, width: int = 800):
        data = self.summarize_compressed_histogram(self.tensor(name))
        return self.render_density(data,
                                   name=name,
                                   line_color=line_color,
                                   range_color=range_color,
                                   levels=levels,
                                   alpha=alpha,
                                   width=width,
                                   height=height)
