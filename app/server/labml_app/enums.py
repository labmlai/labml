class RunEnums:
    RUN_COMPLETED = 'completed'
    RUN_CRASHED = 'crashed'
    RUN_INTERRUPTED = 'interrupted'
    RUN_IN_PROGRESS = 'in progress'
    RUN_UNKNOWN = 'unknown'
    RUN_NOT_RESPONDING = 'no response'


class SeriesEnums:
    GRAD = 'grad'
    PARAM = 'param'
    MODULE = 'module'
    TIME = 'time'
    METRIC = 'metric'
    HYPERPARAMS = 'hp'


class COMPUTEREnums:
    CPU = 'cpu'
    GPU = 'gpu'
    DISK = 'disk'
    MEMORY = 'memory'
    NETWORK = 'net'
    PROCESS = 'process'
    BATTERY = 'battery'


class InsightEnums:
    DANGER = 'danger'
    WARNING = 'warning'
    SUCCESS = 'success'


INDICATORS = [SeriesEnums.GRAD,
              SeriesEnums.PARAM,
              # SeriesEnums.TIME,
              SeriesEnums.MODULE,
              SeriesEnums.METRIC,
              SeriesEnums.HYPERPARAMS]
