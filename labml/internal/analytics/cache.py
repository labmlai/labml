from labml.internal.analytics.indicators import IndicatorClass, Indicator, Run
from labml.internal.analytics.sqlite import SQLiteAnalytics
from labml.internal.analytics.tensorboard import TensorBoardAnalytics

_RUNS = {}

_TENSORBOARD = {}

_SQLITE = {}


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
    except:
        return None

    tensor = tb.tensor(indicator.key)
    if indicator.class_ == IndicatorClass.histogram:
        data = tb.summarize_compressed_histogram(tensor)
    else:
        data = tb.summarize_scalars(tensor)

    return data


def get_sqlite_data(indicator: Indicator):
    run = get_run(indicator.uuid)

    if indicator.uuid not in _SQLITE:
        _SQLITE[indicator.uuid] = SQLiteAnalytics(run.run_info.sqlite_path)

    sqlite: SQLiteAnalytics = _SQLITE[indicator.uuid]

    if indicator.class_ in [IndicatorClass.histogram, IndicatorClass.queue]:
        key = f"{indicator.key}.mean"
    else:
        key = indicator.key

    data = sqlite.scalar(key)
    data = sqlite.summarize_scalars(data)

    return data


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
