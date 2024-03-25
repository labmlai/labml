from typing import Dict, Any

from fastapi import Request
from fastapi.responses import JSONResponse
from labml_db import Model, Index
from labml_db.serializer.pickle import PickleSerializer

from labml_app.logger import logger
from labml_app.enums import COMPUTEREnums
from ..analysis import Analysis
from ..series import SeriesModel, Series
from ..series_collection import SeriesCollection
from ..preferences import Preferences
from ..helper import get_mean_series


@Analysis.db_model(PickleSerializer, 'CPU')
class CPUModel(Model['CPUModel'], SeriesCollection):
    pass


@Analysis.db_index(PickleSerializer, 'cpu_index')
class CPUIndex(Index['CPU']):
    pass


@Analysis.db_model(PickleSerializer, 'cpu_preferences')
class CPUPreferencesModel(Model['CPUPreferencesModel'], Preferences):
    pass


@Analysis.db_index(PickleSerializer, 'cpu_preferences_index')
class CPUPreferencesIndex(Index['CPUPreferences']):
    pass


class CPUAnalysis(Analysis):
    cpu: CPUModel

    def __init__(self, data):
        self.cpu = data

    def track(self, data: Dict[str, SeriesModel]):
        res: Dict[str, SeriesModel] = {}
        for ind, s in data.items():
            ind_type = ind.split('.')[0]
            if ind_type == COMPUTEREnums.CPU:
                res[ind] = s

        self.cpu.track(res, keep_last_24h=True)

    def get_tracking(self):
        res = []
        summary = []
        series_list = []
        for ind, track in self.cpu.tracking.items():
            name = ind.split('.')

            if any(x in ['freq', 'system', 'idle', 'user'] for x in name):
                continue

            series = Series().load(track)
            series_list.append(series.to_data())
            series_data: Dict[str, Any] = series.detail
            series_data['name'] = ''.join(name[-1])

            res.append(series_data)

        if series_list:
            mean_series = Series().load(get_mean_series(series_list)).detail
            mean_series['name'] = 'mean'
            summary = [mean_series]

        if len(res) > 1:
            res.sort(key=lambda s: int(s['name']))

        return res, summary

    @staticmethod
    def get_or_create(session_uuid: str):
        cpu_key = CPUIndex.get(session_uuid)

        if not cpu_key:
            c = CPUModel()
            c.save()
            CPUIndex.set(session_uuid, c.key)

            cp = CPUPreferencesModel()
            cp.save()
            CPUPreferencesIndex.set(session_uuid, cp.key)

            return CPUAnalysis(c)

        return CPUAnalysis(cpu_key.load())

    @staticmethod
    def delete(session_uuid: str):
        cpu_key = CPUIndex.get(session_uuid)
        preferences_key = CPUPreferencesIndex.get(session_uuid)

        if cpu_key:
            c: CPUModel = cpu_key.load()
            CPUIndex.delete(session_uuid)
            c.delete()

        if preferences_key:
            cp: CPUPreferencesModel = preferences_key.load()
            CPUPreferencesIndex.delete(session_uuid)
            cp.delete()


@Analysis.route('GET', 'cpu/{session_uuid}')
async def get_cpu_tracking(request: Request, session_uuid: str) -> Any:
    track_data = []
    summary_data = []
    status_code = 404

    ans = CPUAnalysis.get_or_create(session_uuid)
    if ans:
        track_data, summary_data = ans.get_tracking()
        status_code = 200

    response = JSONResponse({'series': track_data, 'insights': [], 'summary': summary_data})
    response.status_code = status_code

    return response


@Analysis.route('GET', 'cpu/preferences/{session_uuid}')
async def get_cpu_preferences(request: Request, session_uuid: str) -> Any:
    preferences_data = {}

    preferences_key = CPUPreferencesIndex.get(session_uuid)
    if not preferences_key:
        return preferences_data

    cp: CPUPreferencesModel = preferences_key.load()

    return cp.get_data()


@Analysis.route('POST', 'cpu/preferences/{session_uuid}')
async def set_cpu_preferences(request: Request, session_uuid: str) -> Any:
    preferences_key = CPUPreferencesIndex.get(session_uuid)

    if not preferences_key:
        return {}

    cp = preferences_key.load()
    json = await request.json()
    cp.update_preferences(json)

    logger.debug(f'update cpu preferences: {cp.key}')

    return {'errors': cp.errors}
