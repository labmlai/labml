from typing import Dict, Any

from fastapi import Request
from fastapi.responses import JSONResponse
from labml_db import Model, Index
from labml_db.serializer.pickle import PickleSerializer

from labml_app.logger import logger
from labml_app.enums import COMPUTEREnums
from ..analysis import Analysis
from ..series import SeriesModel, Series
from ..preferences import Preferences
from ..series_collection import SeriesCollection


@Analysis.db_model(PickleSerializer, 'GPU')
class GPUModel(Model['GPUModel'], SeriesCollection):
    pass


@Analysis.db_index(PickleSerializer, 'gpu_index')
class GPUIndex(Index['GPU']):
    pass


@Analysis.db_model(PickleSerializer, 'gpu_preferences')
class GPUPreferencesModel(Model['GPUPreferencesModel'], Preferences):
    pass


@Analysis.db_index(PickleSerializer, 'gpu_preferences_index')
class GPUPreferencesIndex(Index['GPUPreferences']):
    pass


class GPUAnalysis(Analysis):
    gpu: GPUModel

    def __init__(self, data):
        self.gpu = data

    def track(self, data: Dict[str, SeriesModel]):
        res: Dict[str, SeriesModel] = {}
        for ind, s in data.items():
            ind_type = ind.split('.')[0]
            if ind_type == COMPUTEREnums.GPU:
                res[ind] = s

        self.gpu.track(res)

    def get_tracking(self):
        res = []
        for ind, track in self.gpu.tracking.items():
            name = ind.split('.')

            if [i for i in name if i in ['total', 'limit']]:
                continue

            series: Dict[str, Any] = Series().load(track).detail
            series['name'] = '.'.join(name[1:])

            res.append(series)

        return res

    @staticmethod
    def get_or_create(session_uuid: str):
        gpu_key = GPUIndex.get(session_uuid)

        if not gpu_key:
            g = GPUModel()
            g.save()
            GPUIndex.set(session_uuid, g.key)

            gp = GPUPreferencesModel()
            gp.save()
            GPUPreferencesIndex.set(session_uuid, gp.key)

            return GPUAnalysis(g)

        return GPUAnalysis(gpu_key.load())

    @staticmethod
    def delete(session_uuid: str):
        gpu_key = GPUIndex.get(session_uuid)
        preferences_key = GPUPreferencesIndex.get(session_uuid)

        if gpu_key:
            g: GPUModel = gpu_key.load()
            GPUIndex.delete(session_uuid)
            g.delete()

        if preferences_key:
            gp: GPUPreferencesModel = preferences_key.load()
            GPUPreferencesIndex.delete(session_uuid)
            gp.delete()


@Analysis.route('GET', 'gpu/{session_uuid}')
async def get_gpu_tracking(request: Request, session_uuid: str) -> Any:
    track_data = []
    status_code = 404

    ans = GPUAnalysis.get_or_create(session_uuid)
    if ans:
        track_data = ans.get_tracking()
        status_code = 200

    response = JSONResponse({'series': track_data, 'insights': []})
    response.status_code = status_code

    return response


@Analysis.route('GET', 'gpu/preferences/{session_uuid}')
async def get_gpu_preferences(request: Request, session_uuid: str) -> Any:
    preferences_data = {}

    preferences_key = GPUPreferencesIndex.get(session_uuid)
    if not preferences_key:
        return preferences_data

    gp: GPUPreferencesModel = preferences_key.load()

    return gp.get_data()


@Analysis.route('POST', 'gpu/preferences/{session_uuid}')
async def set_gpu_preferences(request: Request, session_uuid: str) -> Any:
    preferences_key = GPUPreferencesIndex.get(session_uuid)

    if not preferences_key:
        return {}

    gp = preferences_key.load()
    json = await request.json()
    gp.update_sub_series_preferences(json)

    logger.debug(f'update gpu preferences: {gp.key}')

    return {'errors': gp.errors}
