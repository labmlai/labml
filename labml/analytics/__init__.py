from typing import Tuple, Optional, List, overload

import numpy as np
from labml.internal.analytics import cache as _cache
from labml.internal.analytics.altair import density as _density
from labml.internal.analytics.altair import scatter as _scatter
from labml.internal.analytics.indicators import IndicatorCollection as _IndicatorCollection


class IndicatorCollection(_IndicatorCollection):
    r"""
    You can get a indicator collection with :func:`runs`.

    >>> from labml import analytics
    >>> indicators = analytics.runs('1d3f855874d811eabb9359457a24edc8')

    You can reference individual indicators as attributes.

    >>> train_loss = indicators.train_loss

    You can add multiple indicator collections

    >>> losses = indicators.train_loss + indicators.validation_loss
    """
    pass


def runs(*uuids: str):
    r"""
    This is used to analyze runs.
    It fetches all the log indicators.

    Arguments:
        uuids (str): UUIDs of the runs. You can
            get this from `dashboard <https://github.com/lab-ml/lab_dashboard>`_

    Example:
        >>> from labml import analytics
        >>> indicators = analytics.runs('1d3f855874d811eabb9359457a24edc8')
    """

    indicators = None
    for r in uuids:
        run = _cache.get_run(r)
        indicators = indicators + run.indicators

    return indicators


def get_run(uuid: str):
    r"""
    Returns ``Run`` object
    """
    return _cache.get_run(uuid)


def set_preferred_db(db: str):
    assert db in ['tensorboard', 'sqlite']


@overload
def distribution(indicators: IndicatorCollection, *,
                 levels: int = 5, alpha: int = 0.6,
                 height: int = 400, width: int = 800, height_minimap: int = 100):
    ...


@overload
def distribution(series: List[np.ndarray], names: List[str], *,
                 levels: int = 5, alpha: int = 0.6,
                 height: int = 400, width: int = 800, height_minimap: int = 100):
    ...


@overload
def distribution(series: List[np.ndarray], *,
                 levels: int = 5, alpha: int = 0.6,
                 height: int = 400, width: int = 800, height_minimap: int = 100):
    ...


def distribution(*args: any,
                 levels: int = 5, alpha: int = 0.6,
                 height: int = 400, width: int = 800, height_minimap: int = 100):
    r"""
    Creates a distribution plot distribution with Altair

    :Arguments:
        indicators(IndicatorCollection): Set of indicators to be plotted
        series(List[np.ndarray]): List of series of data
        names(List[str]): List of names of series

    :Keyword Arguments:
        levels: how many levels of the distribution to be plotted
        alpha: opacity of the distribution
        height: height of the visualization
        width: width of the visualization
        height_minimap: height of the view finder

    :Return:
        The Altair visualization

    Example:
        >>> from labml import analytics
        >>> indicators = analytics.runs('1d3f855874d811eabb9359457a24edc8')
        >>> analytics.distribution(indicators)
    """

    series = None
    names = None

    if len(args) == 1:
        if isinstance(args[0], _IndicatorCollection):
            series, names = _cache.get_indicators_data(args[0])
            if not series:
                raise ValueError("No series found")
        elif isinstance(args[0], list):
            series = args[0]
            names = [f'{i + 1}' for i in range(len(series))]
    elif len(args) == 2:
        if isinstance(args[0], list) and isinstance(args[1], list):
            series = args[0]
            names = args[1]

    if series is None:
        raise ValueError("distribution should be called with an indicator collection"
                         " or a series. Check documentation for details.")

    tables = [_density.data_to_table(s) for s in series]

    return _density.render(
        tables,
        names=names,
        levels=levels,
        alpha=alpha,
        width=width,
        height=height,
        height_minimap=height_minimap)


@overload
def scatter(indicators: IndicatorCollection, x_indicators: IndicatorCollection, *,
            noise: Optional[Tuple[float, float]] = None,
            height: int = 400, width: int = 800, height_minimap: int = 100):
    ...


@overload
def scatter(series: List[np.ndarray], names: List[str],
            x_series: np.ndarray, x_name: str, *,
            noise: Optional[Tuple[float, float]] = None,
            height: int = 400, width: int = 800, height_minimap: int = 100):
    ...


@overload
def scatter(series: List[np.ndarray],
            x_series: np.ndarray,
            noise: Optional[Tuple[float, float]] = None,
            height: int = 400, width: int = 800, height_minimap: int = 100):
    ...


def scatter(*args: any,
            noise: Optional[Tuple[float, float]] = None,
            circle_size: int = 20,
            height: int = 400, width: int = 800, height_minimap: int = 100):
    r"""
    Creates a scatter plot with Altair

    :Arguments:
        indicators(IndicatorCollection): Set of indicators to be plotted
        x_indicators(IndicatorCollection): Indicator for x-axis
        series(List[np.ndarray]): List of series of data
        names(List[str]): List of names of series
        x_series(np.ndarray): X series of data
        name(str): Name of X series

    :Keyword Arguments:
        noise: Noise to be added to spread out the scatter plot
        circle_size: size of circles in the plot
        height: height of the visualization
        width: width of the visualization
        height_minimap: height of the view finder

    :Return:
        The Altair visualization

    :Example:
        >>> from labml import analytics
        >>> indicators = analytics.runs('1d3f855874d811eabb9359457a24edc8')
        >>> analytics.scatter(indicators.validation_loss, indicators.train_loss)
    """

    series = None
    names = None
    x_series = None
    x_name = None

    if len(args) == 2:
        if isinstance(args[0], _IndicatorCollection) and isinstance(args[1], _IndicatorCollection):
            series, names = _cache.get_indicators_data(args[0])
            x_series, x_name = _cache.get_indicators_data(args[1])

            if len(x_series) != 1:
                raise ValueError("There should be exactly one series for x-axis")
            if not series:
                raise ValueError("No series found")
            x_series = x_series[0]
            x_name = x_name[0]
        elif isinstance(args[0], list):
            series = args[0]
            names = [f'{i + 1}' for i in range(len(series))]
            x_series = args[1]
            x_name = 'x'
    elif len(args) == 4:
        if isinstance(args[0], list) and isinstance(args[1], list):
            series = args[0]
            names = args[1]
            x_series = args[2]
            x_name = args[3]

    if series is None:
        raise ValueError("scatter should be called with an indicator collection"
                         " or a series. Check documentation for details.")

    tables = [_scatter.data_to_table(s, x_series, noise) for s in series]

    return _scatter.render(
        tables,
        names=names,
        x_name=x_name,
        width=width,
        height=height,
        height_minimap=height_minimap,
        circle_size=circle_size)


def indicator_data(indicators: IndicatorCollection) -> Tuple[List[np.ndarray], List[str]]:
    r"""
    Returns a tuple of a list of series and a list of names of series.
    Each series, `S` is a timeseries of histograms of shape `[T, 10]`,
    where `T` is the number of timesteps.
    `S[:, 0]` is the `global_step`.
    `S[:, 1:10]` represents the distribution at basis points
     `0, 6.68, 15.87, 30.85, 50.00, 69.15, 84.13, 93.32, 100.00`.

    Example:
        >>> from labml import analytics
        >>> indicators = analytics.runs('1d3f855874d811eabb9359457a24edc8')
        >>> analytics.indicator_data(indicators)
    """

    series, names = _cache.get_indicators_data(indicators)

    if not series:
        raise ValueError("No series found")

    return series, names


def artifact_data(indicators: IndicatorCollection) -> Tuple[List[any], List[str]]:
    r"""
    Returns a tuple of a list of series and a list of names of series.
    Each series, ``S`` is a timeseries of histograms of shape ``[T, 10]``,
    where ``T`` is the number of timesteps.
    ``S[:, 0]`` is the `global_step`.
    ``S[:, 1:10]`` represents the distribution at basis points:
    ``0, 6.68, 15.87, 30.85, 50.00, 69.15, 84.13, 93.32, 100.00``.

    Example:
        >>> from labml import analytics
        >>> indicators = analytics.runs('1d3f855874d811eabb9359457a24edc8')
        >>> analytics.artifact_data(indicators)
    """

    series, names = _cache.get_artifacts_data(indicators)

    if not series:
        raise ValueError("No series found")

    return series, names
