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


@Analysis.route('POST', 'logs/std_logger/{run_uuid}')
async def get_std_logger(request: Request, run_uuid: str) -> Any:
    """
            body data: {
                page: int
            }

            page = -2 means get all logs.
            page = -1 means get last page.
            page = n means get nth page.
        """
    json = await request.json()
    page = json.get('page', -1)

    key = StdLoggerIndex.get(run_uuid)
    std_out: StdLoggerModel

    if key is None:
        std_out = StdLoggerModel()
        std_out.save()
        StdLoggerIndex.set(run_uuid, std_out.key)
    else:
        std_out = key.load()

    return std_out.get_data(page_no=page)


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
