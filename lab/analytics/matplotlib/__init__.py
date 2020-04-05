from matplotlib.axes import Axes


class MatPlotLibAnalytics:
    def render_density(self, ax: Axes, data, color, name, *,
                       levels=5,
                       line_width=1,
                       alpha=0.6):
        ln = ax.plot(data[:, 0], data[:, 5],
                     lw=line_width,
                     color=color,
                     alpha=1,
                     label=name)

        for i in range(1, levels):
            ax.fill_between(
                data[:, 0],
                data[:, 5 - i],
                data[:, 5 + i],
                color=color,
                lw=0,
                alpha=alpha ** i)

        return ln
