from typing import Any

from labml_db import Model, Index
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer
from fastapi import Request

from labml_app.analyses.analysis import Analysis
from labml_app.analyses.logs import Logs


class StdLogger(Logs):
    pass


@Analysis.db_model(PickleSerializer, 'std_logger')
class StdLoggerModel(Model['StdLoggerModel'], StdLogger):
    pass


@Analysis.db_index(YamlSerializer, 'std_logger_index.yaml')
class StdLoggerIndex(Index['StdLogger']):
    pass


@Analysis.route('GET', 'logs/std_logger/{run_uuid}')
async def get_std_logger(_: Request, run_uuid: str) -> Any:
    key = StdLoggerIndex.get(run_uuid)
    std_logger: StdLoggerModel

    if key is None:
        std_logger = StdLoggerModel()
        std_logger.save()
        StdLoggerIndex.set(run_uuid, std_logger.key)
    else:
        std_logger = key.load()

    return std_logger.get_data()


def update_std_logger(run_uuid: str, content: str):
    key = StdLoggerIndex.get(run_uuid)
    std_logger: StdLoggerModel

    if key is None:
        std_logger = StdLoggerModel()
        std_logger.save()
        StdLoggerIndex.set(run_uuid, std_logger.key)
    else:
        std_logger = key.load()

    std_logger.update_logs(content)
    std_logger.save()
