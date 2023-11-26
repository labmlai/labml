from typing import Dict, Any

from fastapi import Request
from fastapi.responses import JSONResponse
from labml_db import Model, Index
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer

from labml_app.logger import logger
from labml_app.enums import COMPUTEREnums
from ..analysis import Analysis
from ..series import SeriesModel, Series
from ..series_collection import SeriesCollection
from ..preferences import Preferences
from .. import helper


@Analysis.db_model(PickleSerializer, 'Memory')
class MemoryModel(Model['MemoryModel'], SeriesCollection):
    pass


@Analysis.db_index(YamlSerializer, 'memory_index')
class MemoryIndex(Index['Memory']):
    pass


@Analysis.db_model(PickleSerializer, 'memory_preferences')
class MemoryPreferencesModel(Model['MemoryPreferencesModel'], Preferences):
    pass


@Analysis.db_index(YamlSerializer, 'memory_preferences_index')
class MemoryPreferencesIndex(Index['MemoryPreferences']):
    pass


class MemoryAnalysis(Analysis):
    memory: MemoryModel

    def __init__(self, data):
        self.memory = data

    def track(self, data: Dict[str, SeriesModel]):
        res: Dict[str, SeriesModel] = {}
        for ind, s in data.items():
            ind_type = ind.split('.')[0]
            if ind_type == COMPUTEREnums.MEMORY:
                res[ind] = s

        self.memory.track(res, keep_last_24h=True)

    def get_tracking(self):
        res = []
        for ind, track in self.memory.tracking.items():
            name = ind.split('.')

            if any(x in ['total'] for x in name):
                continue

            series: Dict[str, Any] = Series().load(track).detail
            series['name'] = '.'.join(name)

            res.append(series)

        res.sort(key=lambda s: s['name'])

        helper.remove_common_prefix(res, 'name')

        return res

    @staticmethod
    def get_or_create(session_uuid: str):
        memory_key = MemoryIndex.get(session_uuid)

        if not memory_key:
            m = MemoryModel()
            m.save()
            MemoryIndex.set(session_uuid, m.key)

            mp = MemoryPreferencesModel()
            mp.save()
            MemoryPreferencesIndex.set(session_uuid, mp.key)

            return MemoryAnalysis(m)

        return MemoryAnalysis(memory_key.load())

    @staticmethod
    def delete(session_uuid: str):
        memory_key = MemoryIndex.get(session_uuid)
        preferences_key = MemoryPreferencesIndex.get(session_uuid)

        if memory_key:
            m: MemoryModel = memory_key.load()
            MemoryIndex.delete(session_uuid)
            m.delete()

        if preferences_key:
            mp: MemoryPreferencesModel = preferences_key.load()
            MemoryPreferencesIndex.delete(session_uuid)
            mp.delete()


@Analysis.route('GET', 'memory/{session_uuid}')
async def get_memory_tracking(request: Request, session_uuid: str) -> Any:
    track_data = []
    status_code = 404

    ans = MemoryAnalysis.get_or_create(session_uuid)
    if ans:
        track_data = ans.get_tracking()
        status_code = 200

    response = JSONResponse({'series': track_data, 'insights': []})
    response.status_code = status_code

    return response


@Analysis.route('GET', 'memory/preferences/{session_uuid}')
async def get_memory_preferences(request: Request, session_uuid: str) -> Any:
    preferences_data = {}

    preferences_key = MemoryPreferencesIndex.get(session_uuid)
    if not preferences_key:
        return preferences_data

    mp: MemoryPreferencesModel = preferences_key.load()

    return mp.get_data()


@Analysis.route('POST', 'memory/preferences/{session_uuid}')
async def set_memory_preferences(request: Request, session_uuid: str) -> Any:
    preferences_key = MemoryPreferencesIndex.get(session_uuid)

    if not preferences_key:
        return {}

    mp = preferences_key.load()
    json = await request.json()
    mp.update_preferences(json)

    logger.debug(f'update memory preferences: {mp.key}')

    return {'errors': mp.errors}
