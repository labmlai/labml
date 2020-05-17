import matplotlib.pyplot as plt
import seaborn as sns
from labml.internal.analytics import cache as _cache
from labml.internal.analytics.altair import AltairAnalytics as _AltairAnalytics
from labml.internal.analytics.indicators import IndicatorCollection
from labml.internal.analytics.matplotlib import MatPlotLibAnalytics as _MatPlotLibAnalytics

MATPLOTLIB = _MatPlotLibAnalytics()
ALTAIR = _AltairAnalytics()


def runs(*uuids: str):
    r"""
    This is used to analyze runs.
    It fetches all the log indicators.

    Arguments:
        uuids (str): UUIDs of the runs. You can
            get this from `dashboard <https://github.com/lab-ml/lab_dashboard>`_

    Attributes:
        tb (AltairTensorBoardAnalytics or MatPlotLibTensorBoardAnalytics): analytics
            based on Tensorboard logs

    Example:
        >>> from labml import analytics
        >>> indicators = analytics.runs('1d3f855874d811eabb9359457a24edc8')
    """

    indicators = None
    for r in uuids:
        run = _cache.get_run(r)
        indicators = indicators + run.indicators

    return indicators


def set_preferred_db(db: str):
    assert db in ['tensorboard', 'sqlite']


def get_data(indicators: IndicatorCollection):
    data = {}
    for i, ind in enumerate(indicators):
        data[ind.key] = _cache.get_indicator_data(ind)

    return data


def render_matplotlib(indicators: IndicatorCollection):
    r"""
    Creates a distribution plot distribution with MatPlotLib
    """
    _, ax = plt.subplots(figsize=(18, 9))
    for i, ind in enumerate(indicators):
        data = _cache.get_indicator_data(ind)
        MATPLOTLIB.render_density(ax, data, sns.color_palette()[i], ind.key)
    plt.show()


def render_altair(indicators: IndicatorCollection, *,
                  levels: int = 5, alpha: int = 0.6,
                  height: int = 400, width: int = 800, height_minimap: int = 100):
    r"""
    Creates a distribution plot distribution with Altair

    Arguments:
        indicators(IndicatorCollection): Set of indicators to be plotted

        levels: how many levels of the distribution to be plotted
        alpha: opacity of the distribution
        height: height of the visualization
        width: width of the visualization
        height_minimap: height of the view finder

    Return:
        The Altair visualization

    Example:
        >>> from labml import analytics
        >>> indicators = analytics.runs('1d3f855874d811eabb9359457a24edc8')
        >>> analytics.render_altair(indicators)
    """

    datas = []
    names = []
    for i, ind in enumerate(indicators):
        datas.append(_cache.get_indicator_data(ind))
        names.append(ind.key)

    return ALTAIR.render_density_minimap_multiple(datas,
                                                  names=names,
                                                  levels=levels,
                                                  alpha=alpha,
                                                  width=width,
                                                  height=height,
                                                  height_minimap=height_minimap)
