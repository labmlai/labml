from typing import Any

from starlette.responses import JSONResponse

import labml_app
from labml_db import Model, Index
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer
from fastapi import Request

from labml_app.analyses.analysis import Analysis
from labml_app.analyses.logs import Logs, LogPageType


class StdErr(Logs):
    pass


@Analysis.db_model(PickleSerializer, 'stderr')
class StdErrModel(Model['StdErrModel'], StdErr):
    pass


@Analysis.db_index(YamlSerializer, 'stderr_index.yaml')
class StdErrIndex(Index['StdErr']):
    pass


@Analysis.route('POST', 'logs/stderr/{run_uuid}')
async def get_std_err(request: Request, run_uuid: str) -> Any:
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

    key = StdErrIndex.get(run_uuid)
    std_out: StdErrModel

    if key is None:
        std_out = StdErrModel()
        std_out.save()
        StdErrIndex.set(run_uuid, std_out.key)
    else:
        std_out = key.load()

    return std_out.get_data(page_no=page)


@Analysis.route('POST', 'logs/stderr/{run_uuid}/opt')
async def update_stderr_opt(request: Request, run_uuid: str) -> Any:
    run_uuid = labml_app.db.run.get_main_rank(run_uuid)
    if run_uuid is None:
        return JSONResponse(status_code=404, content={'message': 'Run not found'})

    key = StdErrIndex.get(run_uuid)
    std_err: StdErrModel

    if key is None:
        return JSONResponse(status_code=404, content={'message': 'StdErr not found'})

    std_err = key.load()
    data = await request.json()
    std_err.update_opt(data)
    std_err.save()

    return {'is_successful': True}


def update_stderr(run_uuid: str, content: str):
    key = StdErrIndex.get(run_uuid)
    std_err: StdErrModel

    if key is None:
        std_err = StdErrModel()
        std_err.save()
        StdErrIndex.set(run_uuid, std_err.key)
    else:
        std_err = key.load()

    std_err.update_logs(content)
    std_err.save()
