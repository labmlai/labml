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
from .. import helper


@Analysis.db_model(PickleSerializer, 'Disk')
class DiskModel(Model['DiskModel'], SeriesCollection):
    pass


@Analysis.db_index(PickleSerializer, 'disk_index')
class DiskIndex(Index['Disk']):
    pass


@Analysis.db_model(PickleSerializer, 'disk_preferences')
class DiskPreferencesModel(Model['DiskPreferencesModel'], Preferences):
    pass


@Analysis.db_index(PickleSerializer, 'disk_preferences_index')
class DiskPreferencesIndex(Index['DiskPreferences']):
    pass


class DiskAnalysis(Analysis):
    disk: DiskModel

    def __init__(self, data):
        self.disk = data

    def track(self, data: Dict[str, SeriesModel]):
        res: Dict[str, SeriesModel] = {}
        for ind, s in data.items():
            ind_type = ind.split('.')[0]
            if ind_type == COMPUTEREnums.DISK:
                res[ind] = s

        self.disk.track(res, keep_last_24h=True)

    def get_tracking(self):
        res = []
        for ind, track in self.disk.tracking.items():
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
        disk_key = DiskIndex.get(session_uuid)

        if not disk_key:
            d = DiskModel()
            d.save()
            DiskIndex.set(session_uuid, d.key)

            dp = DiskPreferencesModel()
            dp.save()
            DiskPreferencesIndex.set(session_uuid, dp.key)

            return DiskAnalysis(d)

        return DiskAnalysis(disk_key.load())

    @staticmethod
    def delete(session_uuid: str):
        disk_key = DiskIndex.get(session_uuid)
        preferences_key = DiskPreferencesIndex.get(session_uuid)

        if disk_key:
            d: DiskModel = disk_key.load()
            DiskIndex.delete(session_uuid)
            d.delete()

        if preferences_key:
            dp: DiskPreferencesModel = preferences_key.load()
            DiskPreferencesIndex.delete(session_uuid)
            dp.delete()


@Analysis.route('GET', 'disk/{session_uuid}')
async def get_disk_tracking(request: Request, session_uuid: str) -> Any:
    track_data = []
    status_code = 404

    ans = DiskAnalysis.get_or_create(session_uuid)
    if ans:
        track_data = ans.get_tracking()
        status_code = 200

    response = JSONResponse({'series': track_data, 'insights': []})
    response.status_code = status_code

    return response


@Analysis.route('GET', 'disk/preferences/{session_uuid}')
async def get_disk_preferences(request: Request, session_uuid: str) -> Any:
    preferences_data = {}

    preferences_key = DiskPreferencesIndex.get(session_uuid)
    if not preferences_key:
        return preferences_data

    dp: DiskPreferencesModel = preferences_key.load()

    return dp.get_data()


@Analysis.route('POST', 'disk/preferences/{session_uuid}')
async def set_disk_preferences(request: Request, session_uuid: str) -> Any:
    preferences_key = DiskPreferencesIndex.get(session_uuid)

    if not preferences_key:
        return {}

    dp = preferences_key.load()
    json = await request.json()
    dp.update_preferences(json)

    logger.debug(f'update disk preferences: {dp.key}')

    return {'errors': dp.errors}
