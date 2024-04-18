import base64
from typing import Dict, Any, List, Optional

import numpy as np
from fastapi import Request
from fastapi.responses import JSONResponse
from labml_db import Model, Index, load_keys
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer

from labml_app.logger import logger
from labml_app import utils
from labml_app.enums import INDICATORS
from .distributed_metrics import get_merged_dist_metrics_tracking, set_merged_metrics_preferences, \
    get_merged_metrics_preferences
from ..analysis import Analysis
from ..series import SeriesModel, Series
from ..series_collection import SeriesCollection
from ..preferences import Preferences
from labml_app.settings import INDICATOR_LIMIT
from ...db import run
from ...utils import merge_preferences


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


def mget_preferences(run_uuids: List[str]) -> List[Optional['MetricsPreferencesModel']]:
    run_keys = MetricsPreferencesIndex.mget(run_uuids)
    return load_keys(run_keys)


class MetricsAnalysis(Analysis):
    metrics: MetricsModel

    def __init__(self, data):
        self.metrics = data

    def track(self, data: Dict[str, SeriesModel], run_uuid: str = None):
        res = {}
        current_indicators = list(self.metrics.indicators)
        new_indicators = set()
        for ind, s in data.items():
            ind_split = ind.split('.')
            ind_type = ind_split[0]
            if ind_type not in INDICATORS:
                if ind not in self.metrics.indicators:
                    if len(self.metrics.indicators) >= INDICATOR_LIMIT:
                        continue
                    self.metrics.indicators.add(ind)
                    new_indicators.add(ind)
                res[ind] = s
        if len(new_indicators) > 0:  # update preferences
            try:
                preferences_key = MetricsPreferencesIndex.get(run_uuid)
                mp: MetricsPreferencesModel = preferences_key.load()
                series_preferences = mp.get_data()['series_preferences']

                complete_indicators = current_indicators + list(new_indicators)
                complete_indicators.sort()

                if len(current_indicators) == 0:  # first time
                    series_preferences = utils.get_default_series_preference(complete_indicators)
                else:
                    for i in range(len(complete_indicators)):
                        if complete_indicators[i] in new_indicators:
                            series_preferences.insert(i, -1)

                mp.update_preferences({'series_preferences': series_preferences})
                mp.save()
            except Exception as e:
                logger.error(f'Error updating preferences: {e}')
                raise e

        self.metrics.track(res)

    def get_tracking(self):
        res = []
        is_series_updated = False
        for ind, track in self.metrics.tracking.items():
            name = ind.split('.')

            s = Series().load(track)
            series: Dict[str, Any] = s.to_data()
            series['name'] = '.'.join(name)

            res.append(series)

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
    def get(run_uuid: str) -> Optional['MetricsAnalysis']:
        metrics_key = MetricsIndex.get(run_uuid)

        if not metrics_key:
            return None

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


def get_metrics_tracking_util(track_data: List[Dict[str, Any]], preference_data: List[int],
                              get_all_data: bool):
    filtered_track_data = []
    for preference_item, track_item in zip(preference_data, track_data):
        include_full_data = False

        if get_all_data:
            include_full_data = True
        else:
            include_full_data = preference_item != -1

        filtered_track_data.append(track_item)
        if include_full_data:
            filtered_track_data[-1]['is_summary'] = False
        else:
            filtered_track_data[-1]['value'] = filtered_track_data[-1]['value'][-1:]
            filtered_track_data[-1]['step'] = filtered_track_data[-1]['step'][-1:]
            filtered_track_data[-1]['is_summary'] = True

        s = Series()
        s.update(list(filtered_track_data[-1]['step']), filtered_track_data[-1]['value'])
        details = s.detail
        details['is_summary'] = filtered_track_data[-1]['is_summary']
        details['name'] = filtered_track_data[-1]['name']

        filtered_track_data[-1] = details

    return filtered_track_data


# @utils.mix_panel.MixPanelEvent.time_this(None)
@Analysis.route('POST', 'metrics/{run_uuid}')
async def get_metrics_tracking(request: Request, run_uuid: str) -> Any:
    track_data = []
    status_code = 404

    get_all_data = (await request.json())['get_all']

    #  return merged metrics if applicable
    if len(run_uuid.split('_')) == 1:  # not a rank
        r = run.get(run_uuid)
        if r is not None and r.world_size > 0:  # distributed run
            return get_merged_dist_metrics_tracking(run_uuid, get_all_data)

    run_uuid = utils.get_true_run_uuid(run_uuid)

    track_data = MetricsAnalysis.get_or_create(run_uuid).get_tracking()
    status_code = 200

    preference_data = []
    preferences_key = MetricsPreferencesIndex.get(run_uuid)
    mp: MetricsPreferencesModel
    if preferences_key:
        mp = preferences_key.load()
        preference_data = mp.get_data()['series_preferences']
    else:
        mp = MetricsPreferencesModel()
        mp.save()
        MetricsPreferencesIndex.set(run_uuid, mp.key)

    # update preferences incase it doesn't match with the series
    if len(preference_data) == 0:
        preference_data = utils.get_default_series_preference([s['name'] for s in track_data])
        mp.update_preferences({'series_preferences': preference_data})
        mp.save()
    elif len(preference_data) != len(track_data):
        preference_data = utils.fill_preferences([s['name'] for s in track_data], preference_data)
        mp.update_preferences({'series_preferences': preference_data})
        mp.save()

    preference_data = merge_preferences(run_uuid, preference_data)
    filtered_track_data = get_metrics_tracking_util(track_data, preference_data, get_all_data)

    response = JSONResponse({'series': filtered_track_data, 'insights': []})
    response.status_code = status_code

    return response


@Analysis.route('GET', 'metrics/preferences/{run_uuid}')
async def get_metrics_preferences(request: Request, run_uuid: str) -> Any:
    #  return merged metrics if applicable
    if len(run_uuid.split('_')) == 1:  # not a rank
        r = run.get(run_uuid)
        if r is not None and r.world_size > 0:  # distributed run
            return await get_merged_metrics_preferences(run_uuid)

    run_uuid = utils.get_true_run_uuid(run_uuid)

    preferences_key = MetricsPreferencesIndex.get(run_uuid)
    mp: Optional['MetricsPreferencesModel'] = None
    if preferences_key:
        mp = preferences_key.load()

    if mp is None:
        mp = MetricsPreferencesModel()
        mp.save()
        MetricsPreferencesIndex.set(run_uuid, mp.key)

    return mp.get_data()


@Analysis.route('POST', 'metrics/preferences/{run_uuid}')
async def set_metrics_preferences(request: Request, run_uuid: str) -> Any:
    #  return merged metrics if applicable
    if len(run_uuid.split('_')) == 1:  # not a rank
        r = run.get(run_uuid)
        if r is not None and r.world_size > 0:  # distributed run
            return await set_merged_metrics_preferences(run_uuid, await request.json())

    run_uuid = utils.get_true_run_uuid(run_uuid)

    preferences_key = MetricsPreferencesIndex.get(run_uuid)
    if preferences_key is not None:
        mp = preferences_key.load()
    else:
        mp = MetricsPreferencesModel()
        mp.save()
        MetricsPreferencesIndex.set(run_uuid, mp.key)

    json = await request.json()

    app_preferences = json['series_preferences']
    client_preferences = mp.get_data()['series_preferences']

    if len(app_preferences) != len(client_preferences):
        if 'series_names' not in json:
            raise ValueError('series_names not found in the request')
        series_names = [s['name'] for s in MetricsAnalysis.get_or_create(run_uuid).get_tracking()]
        json['series_preferences'] = (
            utils.update_series_preferences(app_preferences, json['series_names'], series_names))

    mp.update_preferences(json)

    logger.debug(f'update metrics preferences: {mp.key}')

    return {'errors': mp.errors}
