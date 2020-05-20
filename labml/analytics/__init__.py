from typing import Tuple, Optional

from labml.internal.analytics import cache as _cache
from labml.internal.analytics.altair.density import AltairDensity as _AltairDensity
from labml.internal.analytics.altair.scatter import AltairScatter as _AltairScatter
from labml.internal.analytics.indicators import IndicatorCollection

ALTAIR_DENSITY = _AltairDensity()
ALTAIR_SCATTER = _AltairScatter()


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
        d = _cache.get_indicator_data(ind)
        data[ind.key] = d[:, [0, 5]]

    return data


def distribution(indicators: IndicatorCollection, *,
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
        >>> analytics.distribution(indicators)
    """

    datas = []
    names = []
    for i, ind in enumerate(indicators):
        datas.append(_cache.get_indicator_data(ind))
        names.append(ind.key)

    return ALTAIR_DENSITY.render_density_minimap_multiple(
        datas,
        names=names,
        levels=levels,
        alpha=alpha,
        width=width,
        height=height,
        height_minimap=height_minimap)


def scatter(indicators: IndicatorCollection, x: IndicatorCollection, *,
            noise: Optional[Tuple[float, float]] = None,
            height: int = 400, width: int = 800, height_minimap: int = 100):
    r"""
    Creates a distribution plot distribution with Altair

    Arguments:
        indicators(IndicatorCollection): Set of indicators to be plotted
        x(IndicatorCollection): Indicator for x-axis

        noise: Noise to be added to spread out the scatter plot
        height: height of the visualization
        width: width of the visualization
        height_minimap: height of the view finder

    Return:
        The Altair visualization

    Example:
        >>> from labml import analytics
        >>> indicators = analytics.runs('1d3f855874d811eabb9359457a24edc8')
        >>> analytics.distribution(indicators)
    """

    datas = []
    names = []
    x_datas = []
    x_names = []

    for i, ind in enumerate(indicators):
        datas.append(_cache.get_indicator_data(ind))
        names.append(ind.key)
    for i, ind in enumerate(x):
        x_datas.append(_cache.get_indicator_data(ind))
        x_names.append(ind.key)

    assert len(x_datas) == 1

    return ALTAIR_SCATTER.scatter(
        datas, x_datas[0],
        names=names,
        x_name=x_names[0],
        width=width,
        height=height,
        height_minimap=height_minimap,
        noise=noise)
