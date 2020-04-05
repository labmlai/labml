from typing import Optional

from matplotlib.axes import Axes

from . import MatPlotLibAnalytics
from ..sqlite import SQLiteAnalytics


class MatPlotLibSQLiteAnalytics(SQLiteAnalytics, MatPlotLibAnalytics):
    def render_scalar_smoothed(self, name, ax: Axes, color, *,
                               smooth: Optional[int] = 100, levels=5, line_width=1, alpha=0.6):
        data = self.scalar(name)
        data = self.summarize_scalars(data, smooth)
        self.render_density(ax, data, color, name,
                            levels=levels,
                            line_width=line_width,
                            alpha=alpha)

    def render_scalar(self, name, ax: Axes, color, *,
                      line_width=1, alpha=0.6):
        self.render_scalar_smoothed(name, ax, color,
                                    smooth=None, levels=1,
                                    line_width=line_width,
                                    alpha=alpha)
