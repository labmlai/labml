from typing import Any

from labml_db import Model, Index
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer
from fastapi import Request
from starlette.responses import JSONResponse

import labml_app.db.run
from labml_app.analyses.analysis import Analysis
from labml_app.analyses.logs import Logs, LogPageType


class StdOut(Logs):
    pass


@Analysis.db_model(PickleSerializer, 'stdout')
class StdOutModel(Model['StdOutModel'], StdOut):
    pass


@Analysis.db_index(YamlSerializer, 'stdout_index.yaml')
class StdOutIndex(Index['StdOut']):
    pass


@Analysis.route('POST', 'logs/stdout/{run_uuid}')
async def get_stdout(request: Request, run_uuid: str) -> Any:
    """
        body data: {
            page: int
        }

        page = -2 means get all logs.
        page = -1 means get last page.
        page = n means get nth page.
    """
    # get the run

    run_uuid = labml_app.db.run.get_main_rank(run_uuid)
    if run_uuid is None:
        return JSONResponse(status_code=404, content={'message': 'Run not found'})

    json = await request.json()
    page = json.get('page', LogPageType.LAST.value)

    key = StdOutIndex.get(run_uuid)
    std_out: StdOutModel

    if key is None:
        std_out = StdOutModel()
        std_out.save()
        StdOutIndex.set(run_uuid, std_out.key)
    else:
        std_out = key.load()

    return std_out.get_data(page_no=page)


@Analysis.route('POST', 'logs/stdout/{run_uuid}/opt')
async def update_stdout_opt(request: Request, run_uuid: str) -> Any:
    run_uuid = labml_app.db.run.get_main_rank(run_uuid)
    if run_uuid is None:
        return JSONResponse(status_code=404, content={'message': 'Run not found'})

    key = StdOutIndex.get(run_uuid)
    std_out: StdOutModel

    if key is None:
        return JSONResponse(status_code=404, content={'message': 'Stdout not found'})

    std_out = key.load()
    data = await request.json()
    std_out.update_opt(data)
    std_out.save()

    return {'is_successful': True}


def update_stdout(run_uuid: str, content: str):
    key = StdOutIndex.get(run_uuid)
    std_out: StdOutModel

    if key is None:
        std_out = StdOutModel()
        std_out.save()
        StdOutIndex.set(run_uuid, std_out.key)
    else:
        std_out = key.load()

    std_out.update_logs(content)
    std_out.save()
