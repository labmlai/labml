from typing import Any

from labml_db import Model, Index
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer
from requests import Request
from starlette.responses import JSONResponse

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


@Analysis.route('POST', 'logs/std_logger/{run_uuid}')
async def update_std_logger(request: Request, run_uuid: str) -> JSONResponse:
    data = await request.json()
    key = StdLoggerIndex.get(run_uuid)
    std_logger: StdLoggerModel

    if key is None:
        return JSONResponse({'error': 'No logs found'}, status_code=404)
    else:
        std_logger = key.load()

    std_logger.update(data['logs'])

    return JSONResponse(std_logger.get_data())
