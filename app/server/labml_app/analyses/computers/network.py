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


@Analysis.db_model(PickleSerializer, 'Network')
class NetworkModel(Model['NetworkModel'], SeriesCollection):
    pass


@Analysis.db_index(YamlSerializer, 'network_index')
class NetworkIndex(Index['Network']):
    pass


@Analysis.db_model(PickleSerializer, 'network_preferences')
class NetworkPreferencesModel(Model['NetworkPreferencesModel'], Preferences):
    pass


@Analysis.db_index(PickleSerializer, 'network_preferences_index')
class NetworkPreferencesIndex(Index['NetworkPreferences']):
    pass


class NetworkAnalysis(Analysis):
    network: NetworkModel

    def __init__(self, data):
        self.network = data

    def track(self, data: Dict[str, SeriesModel]):
        res: Dict[str, SeriesModel] = {}
        for ind, s in data.items():
            ind_type = ind.split('.')[0]
            if ind_type == COMPUTEREnums.NETWORK:
                res[ind] = s

        self.network.track(res, keep_last_24h=True)

    def get_tracking(self):
        res = []
        for ind, track in self.network.tracking.items():
            name = ind.split('.')
            series: Dict[str, Any] = Series().load(track).detail
            series['name'] = '.'.join(name)

            res.append(series)

        res.sort(key=lambda s: s['name'])

        helper.remove_common_prefix(res, 'name')

        return res

    @staticmethod
    def get_or_create(session_uuid: str):
        network_key = NetworkIndex.get(session_uuid)

        if not network_key:
            n = NetworkModel()
            n.save()
            NetworkIndex.set(session_uuid, n.key)

            np = NetworkPreferencesModel()
            np.save()
            NetworkPreferencesIndex.set(session_uuid, np.key)

            return NetworkAnalysis(n)

        return NetworkAnalysis(network_key.load())

    @staticmethod
    def delete(session_uuid: str):
        network_key = NetworkIndex.get(session_uuid)
        preferences_key = NetworkPreferencesIndex.get(session_uuid)

        if network_key:
            n: NetworkModel = network_key.load()
            NetworkIndex.delete(session_uuid)
            n.delete()

        if preferences_key:
            np: NetworkPreferencesModel = preferences_key.load()
            NetworkPreferencesIndex.delete(session_uuid)
            np.delete()


@Analysis.route('GET', 'network/{session_uuid}')
async def get_network_tracking(request: Request, session_uuid: str) -> Any:
    track_data = []
    status_code = 404

    ans = NetworkAnalysis.get_or_create(session_uuid)
    if ans:
        track_data = ans.get_tracking()
        status_code = 200

    response = JSONResponse({'series': track_data, 'insights': []})
    response.status_code = status_code

    return response


@Analysis.route('GET', 'network/preferences/{session_uuid}')
async def get_network_preferences(request: Request, session_uuid: str) -> Any:
    preferences_data = {}

    preferences_key = NetworkPreferencesIndex.get(session_uuid)
    if not preferences_key:
        return preferences_data

    np: NetworkPreferencesModel = preferences_key.load()

    return np.get_data()


@Analysis.route('POST', 'network/preferences/{session_uuid}')
async def set_network_preferences(request: Request, session_uuid: str) -> Any:
    preferences_key = NetworkPreferencesIndex.get(session_uuid)

    if not preferences_key:
        return {}

    np = preferences_key.load()
    json = await request.json()
    np.update_preferences(json)

    logger.debug(f'update network preferences: {np.key}')

    return {'errors': np.errors}
