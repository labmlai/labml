from typing import List

from . import AltairAnalytics
from ..tensorboard import TensorBoardAnalytics


class AltairTensorBoardAnalytics(TensorBoardAnalytics, AltairAnalytics):
    def render_scalar(self, name, *,
                      line_color='steelblue', range_color='steelblue',
                      levels: int = 5, alpha: float = 0.6,
                      points: int = 100,
                      height: int = 400, width: int = 800, height_minimap: int = 100):
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
