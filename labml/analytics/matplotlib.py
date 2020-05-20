import matplotlib.pyplot as plt
import seaborn as sns

from labml.internal.analytics import cache as _cache
from labml.internal.analytics.indicators import IndicatorCollection
from labml.internal.analytics.matplotlib import MatPlotLibAnalytics as _MatPlotLibAnalytics

MATPLOTLIB = _MatPlotLibAnalytics()


def render_matplotlib(indicators: IndicatorCollection):
    r"""
    Creates a distribution plot distribution with MatPlotLib
    """
    _, ax = plt.subplots(figsize=(18, 9))
    for i, ind in enumerate(indicators):
        data = _cache.get_indicator_data(ind)
        MATPLOTLIB.render_density(ax, data, sns.color_palette()[i], ind.key)
    plt.show()
