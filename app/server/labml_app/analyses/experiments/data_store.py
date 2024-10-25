from typing import Dict, Any

from labml_db import Index, Model
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer

from labml_app.analyses.analysis import Analysis


@Analysis.db_model(PickleSerializer, 'data_store')
class DataStoreModel(Model['data_store']):
    data: Dict[str, Any]

    @classmethod
    def defaults(cls):
        return dict(
            data={}
        )

    def get_data(self):
        return self.data

    def update_data(self, data: Dict[str, Any]):
        for key, value in data.items():
            self.data[key] = value


@Analysis.db_index(YamlSerializer, 'datastore_index.yaml')
class DataStoreIndex(Index['data_store']):
    pass


@Analysis.route('GET', 'datastore/{run_uuid}')
async def get_data_store(run_uuid: str) -> Any:
    key = DataStoreIndex.get(run_uuid)
    if key is None:
        data_store = DataStoreModel()
        data_store.save()
        DataStoreIndex.set(run_uuid, data_store.key)

        return data_store.get_data()
    else:
        return key.load().get_data()


@Analysis.route('POST', 'datastore/{run_uuid}')
async def update_data_store(run_uuid: str, data: Dict[str, Any]) -> Any:
    key = DataStoreIndex.get(run_uuid)
    if key is None:
        data_store = DataStoreModel()
        data_store.save()
        DataStoreIndex.set(run_uuid, data_store.key)
    else:
        data_store = key.load()

    data_store.update_data(data)
    data_store.save()

    return data_store.get_data()
