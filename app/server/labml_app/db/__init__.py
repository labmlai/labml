import os
from pathlib import Path

from labml_db import Model, Index
from labml_db.driver.redis import RedisDbDriver
from labml_db.driver.file import FileDbDriver
from labml_db.index_driver.redis import RedisIndexDbDriver
from labml_db.index_driver.file import FileIndexDbDriver
from labml_db.serializer.json import JsonSerializer
from labml_db.serializer.yaml import YamlSerializer
from labml_db.serializer.pickle import PickleSerializer

from .. import settings
from . import project
from . import user
from . import status
from . import app_token
from . import run
from . import session
from . import computer
from . import job
from . import blocked_uuids
from .. import analyses

Models = [(YamlSerializer(), user.User),
          (YamlSerializer(), project.Project),
          (JsonSerializer(), status.Status),
          (JsonSerializer(), status.RunStatus),
          (JsonSerializer(), app_token.AppToken),
          (JsonSerializer(), run.Run),
          (JsonSerializer(), session.Session),
          (PickleSerializer(), job.Job),
          (PickleSerializer(), computer.Computer)] + [(s(), m) for s, m, p in analyses.AnalysisManager.get_db_models()]

Indexes = [project.ProjectIndex,
           user.UserIndex,
           blocked_uuids.BlockedRunIndex,
           blocked_uuids.BlockedSessionIndex,
           user.TokenOwnerIndex,
           app_token.AppTokenIndex,
           run.RunIndex,
           session.SessionIndex,
           job.JobIndex,
           computer.ComputerIndex] + [m for s, m, p in analyses.AnalysisManager.get_db_indexes()]


def get_data_path():
    package_path = Path(os.path.dirname(os.path.abspath(__file__))).parent

    data_path = package_path / 'data'
    if not data_path.exists():
        raise RuntimeError(f'Data folder not found. Package path: {str(package_path)}')

    return data_path


def init_db():
    data_path = get_data_path()

    if settings.IS_LOCAL_SETUP:
        Model.set_db_drivers(
            [FileDbDriver(PickleSerializer(), m, Path(f'{data_path}/{m.__name__}')) for s, m in Models])
        Index.set_db_drivers(
            [FileIndexDbDriver(YamlSerializer(), m, Path(f'{data_path}/{m.__name__}.yaml')) for m in Indexes])
    else:
        import redis
        db = redis.Redis(host='localhost', port=6379, db=0)

        Model.set_db_drivers([RedisDbDriver(s, m, db) for s, m in Models])
        Index.set_db_drivers([RedisIndexDbDriver(m, db) for m in Indexes])

    project.create_project(settings.FLOAT_PROJECT_TOKEN, 'float project')
    project.create_project(settings.SAMPLES_PROJECT_TOKEN, 'samples project')
