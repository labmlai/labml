from typing import Any

from labml_db import Model, Index
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer
from fastapi import Request

from labml_app.analyses.analysis import Analysis
from labml_app.analyses.logs import Logs


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
    json = await request.json()
    page = json.get('page', -1)

    key = StdOutIndex.get(run_uuid)
    std_out: StdOutModel

    if key is None:
        std_out = StdOutModel()
        std_out.save()
        StdOutIndex.set(run_uuid, std_out.key)
    else:
        std_out = key.load()

    return std_out.get_data(page_no=page)


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
