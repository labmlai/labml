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


@Analysis.db_model(PickleSerializer, 'Battery')
class BatteryModel(Model['BatteryModel'], SeriesCollection):
    pass


@Analysis.db_index(PickleSerializer, 'battery_index')
class BatteryIndex(Index['Battery']):
    pass


@Analysis.db_model(PickleSerializer, 'battery_preferences')
class BatteryPreferencesModel(Model['BatteryPreferencesModel'], Preferences):
    pass


@Analysis.db_index(PickleSerializer, 'battery_preferences_index')
class BatteryPreferencesIndex(Index['BatteryPreferences']):
    pass


class BatteryAnalysis(Analysis):
    battery: BatteryModel

    def __init__(self, data):
        self.battery = data

    def track(self, data: Dict[str, SeriesModel]):
        res: Dict[str, SeriesModel] = {}
        for ind, s in data.items():
            ind_type = ind.split('.')[0]
            if ind_type == COMPUTEREnums.BATTERY:
                res[ind] = s

        self.battery.track(res, keep_last_24h=True)

    def get_tracking(self):
        res = []
        summary = None
        for ind, track in self.battery.tracking.items():
            name = ind.split('.')

            if 'secsleft' in name:
                continue

            series: Dict[str, Any] = Series().load(track).detail
            series['name'] = ''.join(name[-1])

            if series['name'] == 'percent':
                summary = series

            res.append(series)

        if res:
            summary = [summary]

        if len(res) > 1:
            res.sort(key=lambda s: s['name'])

        if summary is None:
            summary = []

        return res, summary

    @staticmethod
    def get_or_create(session_uuid: str):
        battery_key = BatteryIndex.get(session_uuid)

        if not battery_key:
            b = BatteryModel()
            b.save()
            BatteryIndex.set(session_uuid, b.key)

            bp = BatteryPreferencesModel()
            bp.save()
            BatteryPreferencesIndex.set(session_uuid, bp.key)

            return BatteryAnalysis(b)

        return BatteryAnalysis(battery_key.load())

    @staticmethod
    def delete(session_uuid: str):
        battery_key = BatteryIndex.get(session_uuid)
        preferences_key = BatteryPreferencesIndex.get(session_uuid)

        if battery_key:
            b: BatteryModel = battery_key.load()
            BatteryIndex.delete(session_uuid)
            b.delete()

        if preferences_key:
            bp: BatteryPreferencesModel = preferences_key.load()
            BatteryPreferencesIndex.delete(session_uuid)
            bp.delete()


@Analysis.route('GET', 'battery/{session_uuid}')
async def get_battery_tracking(request: Request, session_uuid: str) -> Any:
    track_data = []
    summary_data = []
    status_code = 404

    ans = BatteryAnalysis.get_or_create(session_uuid)
    if ans:
        track_data, summary_data = ans.get_tracking()
        status_code = 200

    response = JSONResponse({'series': track_data, 'insights': [], 'summary': summary_data})
    response.status_code = status_code

    return response


@Analysis.route('GET', 'battery/preferences/{session_uuid}')
async def get_battery_preferences(request: Request, session_uuid: str) -> Any:
    preferences_data = {}

    preferences_key = BatteryPreferencesIndex.get(session_uuid)
    if not preferences_key:
        return preferences_data

    bp: BatteryPreferencesModel = preferences_key.load()

    return bp.get_data()


@Analysis.route('POST', 'battery/preferences/{session_uuid}')
async def set_battery_preferences(request: Request, session_uuid: str) -> Any:
    preferences_key = BatteryPreferencesIndex.get(session_uuid)

    if not preferences_key:
        return {}

    bp = preferences_key.load()
    json = await request.json()
    bp.update_preferences(json)

    logger.debug(f'update battery_key preferences: {bp.key}')

    return {'errors': bp.errors}
