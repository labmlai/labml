from typing import Dict, Any, List, Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from labml_db import Model, Index, load_keys
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer

from labml_app.logger import logger
from labml_app import utils
from labml_app.enums import INDICATORS
from ..analysis import Analysis
from ..series import SeriesModel, Series
from ..series_collection import SeriesCollection
from ..preferences import Preferences
from labml_app.settings import INDICATOR_LIMIT


@Analysis.db_model(PickleSerializer, 'metrics')
class MetricsModel(Model['MetricsModel'], SeriesCollection):
    pass


@Analysis.db_model(PickleSerializer, 'metrics_preferences')
class MetricsPreferencesModel(Model['MetricsPreferencesModel'], Preferences):
    pass


@Analysis.db_index(YamlSerializer, 'metrics_preferences_index.yaml')
class MetricsPreferencesIndex(Index['MetricsPreferences']):
    pass


@Analysis.db_index(YamlSerializer, 'metrics_index.yaml')
class MetricsIndex(Index['Metrics']):
    pass


def mget(run_uuids: List[str]) -> List[Optional['MetricsAnalysis']]:
    run_keys = MetricsIndex.mget(run_uuids)
    return load_keys(run_keys)


class MetricsAnalysis(Analysis):
    metrics: MetricsModel

    def __init__(self, data):
        self.metrics = data

    def track(self, data: Dict[str, SeriesModel]):
        res = {}
        for ind, s in data.items():
            ind_split = ind.split('.')
            ind_type = ind_split[0]
            if ind_type not in INDICATORS:
                if ind not in self.metrics.indicators:
                    if len(self.metrics.indicators) >= INDICATOR_LIMIT:
                        continue
                    self.metrics.indicators.add('.'.join(ind))

                res[ind] = s

        self.metrics.track(res)

    def get_tracking(self):
        res = []
        is_series_updated = False
        for ind, track in self.metrics.tracking.items():
            name = ind.split('.')

            s = Series().load(track)
            series: Dict[str, Any] = s.detail
            series['name'] = '.'.join(name)

            if s.is_smoothed_updated:
                self.metrics.tracking[ind] = s.to_data()
                is_series_updated = True

            res.append(series)

        if is_series_updated:
            self.metrics.save()

        res.sort(key=lambda s: s['name'])

        return res

    @staticmethod
    def get_or_create(run_uuid: str):
        metrics_key = MetricsIndex.get(run_uuid)

        if not metrics_key:
            m = MetricsModel()
            m.save()
            MetricsIndex.set(run_uuid, m.key)

            mp = MetricsPreferencesModel()
            mp.save()
            MetricsPreferencesIndex.set(run_uuid, mp.key)

            return MetricsAnalysis(m)

        return MetricsAnalysis(metrics_key.load())

    @staticmethod
    def delete(run_uuid: str):
        metrics_key = MetricsIndex.get(run_uuid)
        preferences_key = MetricsPreferencesIndex.get(run_uuid)

        if metrics_key:
            m: MetricsModel = metrics_key.load()
            MetricsIndex.delete(run_uuid)
            m.delete()

        if preferences_key:
            mp: MetricsPreferencesModel = preferences_key.load()
            MetricsPreferencesIndex.delete(run_uuid)
            mp.delete()


# @utils.mix_panel.MixPanelEvent.time_this(None)
@Analysis.route('GET', 'metrics/{run_uuid}')
async def get_metrics_tracking(request: Request, run_uuid: str) -> Any:
    track_data = []
    status_code = 404

    # TODO temporary change to used run_uuid as rank 0
    run_uuid = utils.get_true_run_uuid(run_uuid)

    ans = MetricsAnalysis.get_or_create(run_uuid)
    if ans:
        track_data = ans.get_tracking()
        status_code = 200

    response = JSONResponse({'series': track_data, 'insights': []})
    response.status_code = status_code

    return response


@Analysis.route('GET', 'metrics/preferences/{run_uuid}')
async def get_metrics_preferences(request: Request, run_uuid: str) -> Any:
    preferences_data = {}

    # TODO temporary change to used run_uuid as rank 0
    run_uuid = utils.get_true_run_uuid(run_uuid)

    preferences_key = MetricsPreferencesIndex.get(run_uuid)
    if not preferences_key:
        return preferences_data

    mp: MetricsPreferencesModel = preferences_key.load()

    return mp.get_data()


@Analysis.route('POST', 'metrics/preferences/{run_uuid}')
async def set_metrics_preferences(request: Request, run_uuid: str) -> Any:
    # TODO temporary change to used run_uuid as rank 0
    run_uuid = utils.get_true_run_uuid(run_uuid)

    preferences_key = MetricsPreferencesIndex.get(run_uuid)

    if not preferences_key:
        return {}

    mp = preferences_key.load()
    json = await request.json()
    mp.update_preferences(json)

    logger.debug(f'update metrics preferences: {mp.key}')

    return {'errors': mp.errors}
