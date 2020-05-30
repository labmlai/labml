from pathlib import PurePath

import numpy as np
from labml.internal.analytics.indicators import IndicatorClass, Indicator, Run
from labml.internal.analytics.sqlite import SQLiteAnalytics
from labml.internal.analytics.tensorboard import TensorBoardAnalytics

_RUNS = {}

_TENSORBOARD = {}

_SQLITE = {}

_NUMPY_ARRAYS = {}


def get_run(uuid: str) -> Run:
    if uuid not in _RUNS:
        _RUNS[uuid] = Run(uuid)

    return _RUNS[uuid]


def get_tensorboard_data(indicator: Indicator):
    run = get_run(indicator.uuid)

    if indicator.uuid not in _TENSORBOARD:
        _TENSORBOARD[indicator.uuid] = TensorBoardAnalytics(run.run_info.tensorboard_log_path)

    tb: TensorBoardAnalytics = _TENSORBOARD[indicator.uuid]
    try:
        tb.load()
    except FileNotFoundError:
        return None

    try:
        tensor = tb.tensor(indicator.key)
    except KeyError:
        return None
    if indicator.class_ in [IndicatorClass.histogram, IndicatorClass.queue]:
        data = tb.summarize_compressed_histogram(tensor)
    else:
        data = tb.summarize_scalars(tensor)

    return data


def _get_sqlite_scalar_data(sqlite: SQLiteAnalytics, key: str):
    data = sqlite.scalar(key)
    if not data:
        return None

    data = sqlite.summarize_scalars(data)

    return data


def get_sqlite_data(indicator: Indicator):
    run = get_run(indicator.uuid)

    if indicator.uuid not in _SQLITE:
        _SQLITE[indicator.uuid] = SQLiteAnalytics(run.run_info.sqlite_path)

    sqlite: SQLiteAnalytics = _SQLITE[indicator.uuid]

    if indicator.class_ in [IndicatorClass.histogram, IndicatorClass.queue]:
        return _get_sqlite_scalar_data(sqlite, f"{indicator.key}.mean")
    elif indicator.class_ == IndicatorClass.scalar:
        return _get_sqlite_scalar_data(sqlite, indicator.key)
    else:
        return None


_PREFERRED_DB = 'tensorboard'


def set_preferred_db(db: str):
    global _PREFERRED_DB
    _PREFERRED_DB = db


def get_indicator_data(indicator: Indicator):
    if _PREFERRED_DB == 'tensorboard':
        data = get_tensorboard_data(indicator)
        if data is None:
            data = get_sqlite_data(indicator)
    else:
        data = get_sqlite_data(indicator)
        if data is None:
            data = get_tensorboard_data(indicator)

    return data


def _get_numpy_array(path: PurePath):
    path = str(path)
    if path not in _NUMPY_ARRAYS:
        _NUMPY_ARRAYS[path] = np.load(path)

    return _NUMPY_ARRAYS[path]


def get_artifact_data(indicator: Indicator):
    run = get_run(indicator.uuid)

    if indicator.uuid not in _SQLITE:
        _SQLITE[indicator.uuid] = SQLiteAnalytics(run.run_info.sqlite_path)

    sqlite: SQLiteAnalytics = _SQLITE[indicator.uuid]

    if indicator.class_ != IndicatorClass.tensor:
        return None

    data = sqlite.tensor(indicator.key)

    if not data:
        return None

    data = [(c[0], _get_numpy_array(run.run_info.artifacts_folder / c[1])) for c in data]

    return data
