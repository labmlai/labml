from typing import Any

from starlette.responses import JSONResponse

import labml_app
from labml_db import Model, Index
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer
from fastapi import Request

from labml_app.analyses.analysis import Analysis
from labml_app.analyses.logs import Logs, LogPageType


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
    run_uuid = labml_app.db.run.get_main_rank(run_uuid)
    if run_uuid is None:
        return JSONResponse(status_code=404, content={'message': 'Run not found'})

    json = await request.json()
    page = json.get('page', LogPageType.LAST.value)

    key = StdLoggerIndex.get(run_uuid)
    std_out: StdLoggerModel

    if key is None:
        std_out = StdLoggerModel()
        std_out.save()
        StdLoggerIndex.set(run_uuid, std_out.key)
    else:
        std_out = key.load()

    return std_out.get_data(page_no=page)


@Analysis.route('POST', 'logs/std_logger/{run_uuid}/opt')
async def update_stdlogger_opt(request: Request, run_uuid: str) -> Any:
    run_uuid = labml_app.db.run.get_main_rank(run_uuid)
    if run_uuid is None:
        return JSONResponse(status_code=404, content={'message': 'Run not found'})

    key = StdLoggerIndex.get(run_uuid)
    std_logger: StdLoggerModel

    if key is None:
        return JSONResponse(status_code=404, content={'message': 'StdLogger not found'})

    std_logger = key.load()
    data = await request.json()
    std_logger.update_opt(data)
    std_logger.save()

    return {'is_successful': True}


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
