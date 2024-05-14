import os
from typing import List, Type, Optional
import pickle as pkl

from labml_db.model import ModelDict
from labml_db import Model, Index
from labml_db.driver.mongo import MongoDbDriver
from labml_db.index_driver.mongo import MongoIndexDbDriver

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from .. import settings
from . import project, folder
from . import user
from . import status
from . import app_token
from . import run
from . import session
from . import computer
from .. import analyses


class MongoPickleDbDriver(MongoDbDriver):
    def __init__(self, model_cls: Type['Model'], db: 'pymongo.mongo_client.database.Database'):
        super().__init__(model_cls, db)

    @staticmethod
    def _from_dump(data: pkl.BINBYTES) -> 'ModelDict':
        if data is not None:
            data = pkl.loads(data['data'])

        return data

    @staticmethod
    def _to_dump(data: 'ModelDict'):
        return {'data': pkl.dumps(data)}

    def save_dict(self, key: str, data: 'ModelDict'):
        data = self._to_dump(data)
        return super().save_dict(key, data)

    def load_dict(self, key: str) -> Optional[ModelDict]:
        data = super().load_dict(key)

        return self._from_dump(data)

    def mload_dict(self, keys: List[str]) -> List[Optional[ModelDict]]:
        data = super().mload_dict(keys)

        return [self._from_dump(d) for d in data]

    def msave_dict(self, keys: List[str], data: List[ModelDict]):
        data = [self._to_dump(d) for d in data]

        return super().msave_dict(keys, data)


models = [user.User,
          project.Project,
          folder.Folder,
          status.Status,
          status.RunStatus,
          app_token.AppToken,
          run.Run,
          session.Session,
          computer.Computer] + [m for s, m, p in analyses.AnalysisManager.get_db_models()]

indexes = [project.ProjectIndex,
           user.UserIndex,
           user.UserEmailIndex,
           user.UserTokenIndex,
           user.UserSessionTokenIndex,
           user.TokenOwnerIndex,
           app_token.AppTokenIndex,
           run.RunIndex,
           session.SessionIndex,
           computer.ComputerIndex] + [m for s, m, p in analyses.AnalysisManager.get_db_indexes()]


def init_mongo_db(mongo_address: str = '', port: int = 27017):
    if not mongo_address:
        if 'MONGO_HOST' in os.environ:
            mongo_address = os.getenv('MONGO_HOST')
        else:
            mongo_address = 'localhost'

    mongo_client = MongoClient(host=mongo_address, port=port, connect=False)

    try:
        mongo_client.admin.command('ismaster')
    except ConnectionFailure:
        raise Exception(
            f'MongoDB is either not installed or not running on the {mongo_address} at port {port}. '
            'For detailed instructions on installation, please refer to the tutorial available at '
            'https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/.'
        )

    db = mongo_client['labml']

    Model.set_db_drivers([MongoPickleDbDriver(m, db) for m in models])
    Index.set_db_drivers([MongoIndexDbDriver(m, db) for m in indexes])

    project.create_project(settings.FLOAT_PROJECT_TOKEN, 'float project')
    project.create_project(settings.SAMPLES_PROJECT_TOKEN, 'samples project')

    return mongo_client
