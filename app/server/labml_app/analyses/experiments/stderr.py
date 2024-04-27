from typing import Any

from labml_db import Model, Index
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer
from fastapi import Request

from labml_app.analyses.analysis import Analysis
from labml_app.analyses.logs import Logs


class StdErr(Logs):
    pass


@Analysis.db_model(PickleSerializer, 'stderr')
class StdErrModel(Model['StdErrModel'], StdErr):
    pass


@Analysis.db_index(YamlSerializer, 'stderr_index.yaml')
class StdErrIndex(Index['StdErr']):
    pass


@Analysis.route('GET', 'logs/stderr/{run_uuid}')
async def get_std_err(_: Request, run_uuid: str) -> Any:
    key = StdErrIndex.get(run_uuid)
    std_err: StdErrModel

    if key is None:
        std_err = StdErrModel()
        std_err.save()
        StdErrIndex.set(run_uuid, std_err.key)
    else:
        std_err = key.load()

    return std_err.get_data()


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
