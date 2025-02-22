from typing import Dict, Any
import yaml

from labml_db import Index, Model
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer
from starlette.responses import JSONResponse

import labml_app.db.run as run
import labml_app.db.dist_run as dist_run
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
        return {
            'yaml_string': yaml.dump(self.data),
            'dictionary': self.data
        }

    def set_data(self, data: Dict[str, Any]):
        self.data = data


@Analysis.db_index(YamlSerializer, 'datastore_index.yaml')
class DataStoreIndex(Index['data_store']):
    pass


@Analysis.route('GET', 'datastore/{run_uuid}')
async def get_data_store(run_uuid: str) -> Any:
    dr = dist_run.get(run_uuid)
    if dr is None:
        r = run.get(run_uuid)
        if r is None:
            return JSONResponse({'error': 'Run not found'}, status_code=404)
        if r.parent_run_uuid != "":
            run_uuid = r.parent_run_uuid

    key = DataStoreIndex.get(run_uuid)
    if key is None:
        data_store = DataStoreModel()
        data_store.save()
        DataStoreIndex.set(run_uuid, data_store.key)

        return JSONResponse(data_store.get_data(), status_code=201)
    else:
        return JSONResponse(key.load().get_data(), status_code=200)


@Analysis.route('POST', 'datastore/{run_uuid}')
async def update_data_store(run_uuid: str, data: Dict[str, Any]) -> Any:
    try:
        data_dict = yaml.safe_load(data.get('yaml_string', ""))
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)

    if data_dict is None:
        return JSONResponse({'error': 'Attempting to set empty dictionary'}, status_code=400)

    dr = dist_run.get(run_uuid)
    if dr is None:
        r = run.get(run_uuid)
        if r is None:
            return JSONResponse({'error': 'Run not found'}, status_code=404)
        if r.parent_run_uuid != "":
            run_uuid = r.parent_run_uuid

    key = DataStoreIndex.get(run_uuid)
    if key is None:
        data_store = DataStoreModel()
        data_store.save()
        DataStoreIndex.set(run_uuid, data_store.key)
    else:
        data_store = key.load()

    data_store.set_data(data_dict)
    data_store.save()

    return JSONResponse(data_store.get_data(), status_code=200)
