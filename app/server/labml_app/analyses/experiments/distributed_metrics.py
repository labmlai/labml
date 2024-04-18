import base64
from typing import Any, Optional, Dict, List

import numpy as np
from fastapi import Request
from fastapi.responses import JSONResponse
from labml_db import Model, Index, load_keys
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer

from labml_app import utils
from ..analysis import Analysis
from . import metrics
from ..preferences import Preferences
from ..series import Series
from ...logger import logger
from ...db import run
from ...utils import get_default_series_preference, fill_preferences, merge_preferences


@Analysis.db_model(PickleSerializer, 'merged_metrics_preferences')
class DistMetricsPreferencesModel(Model['DistMetricsPreferencesModel'], Preferences):
    pass


@Analysis.db_index(YamlSerializer, 'merged_metrics_preferences_index.yaml')
class DistMetricsPreferencesIndex(Index['DistMetricsPreferences']):
    pass


def mget_preferences(run_uuids: List[str]) -> List[Optional['DistMetricsPreferencesModel']]:
    run_keys = DistMetricsPreferencesIndex.mget(run_uuids)
    return load_keys(run_keys)


async def get_merged_metrics_preferences(run_uuid: str) -> Any:
    preferences_key = DistMetricsPreferencesIndex.get(run_uuid)
    if not preferences_key:
        mp = DistMetricsPreferencesModel()
        mp.save()
        preferences_key = mp.key

    mp: DistMetricsPreferencesModel = preferences_key.load()

    return mp.get_data()


async def set_merged_metrics_preferences(run_uuid: str, data: Dict[str, any]) -> Any:
    preferences_key = DistMetricsPreferencesIndex.get(run_uuid)

    mp = None
    if not preferences_key:
        mp = DistMetricsPreferencesModel()
        mp.save()
        DistMetricsPreferencesIndex.set(run_uuid, mp.key)

    if not mp:
        mp = preferences_key.load()

    app_preferences = data['series_preferences']
    client_preferences = mp.get_data()['series_preferences']

    if len(app_preferences) != len(client_preferences):
        if 'series_names' not in data:
            raise ValueError('series_names not found in the request')
        series_names = [s['name'] for s in metrics.MetricsAnalysis.get_or_create(run_uuid).get_tracking()]
        data['series_preferences'] = (
            utils.update_series_preferences(app_preferences, data['series_names'], series_names))

    mp.update_preferences(data)

    logger.debug(f'update distributed metrics preferences: {mp.key}')

    return {'errors': mp.errors}


def _dist_series_merge(step_list: List, value_list: List) -> Dict[str, Any]:
    length = max([len(v) for v in value_list if v is not None])
    num_series = len(step_list)

    steps = []
    values = []
    for i in range(length):
        value_sum = 0
        step_sum = 0
        count = 0
        for j in range(num_series):
            if i >= len(value_list[j]):
                continue
            value_sum += value_list[j][i]
            step_sum += step_list[j][i]
            count += 1
        steps.append(step_sum / count)
        values.append(value_sum / count)

    s = Series()
    s.update(list(steps), list(values))
    details = s.detail

    return details


def get_merged_metric_tracking_util(track_data_list, preference_data, get_all_data: bool):
    series_names = []
    series_list = {}
    for track_data in track_data_list:
        for track_item in track_data:
            if track_item['name'] not in series_list:
                series_list[track_item['name']] = {'step': [], 'value': []}
                series_names.append(track_item['name'])

            series_list[track_item['name']]['step'].append(track_item['step'])
            series_list[track_item['name']]['value'].append(track_item['value'])

    filtered_track_data = []
    for preference_item, series_name in zip(preference_data, series_names):
        if get_all_data:
            include_full_data = True
        else:
            include_full_data = preference_item != -1

        if include_full_data:
            step_list = series_list[series_name]['step']
            value_list = series_list[series_name]['value']

            s = _dist_series_merge(step_list, value_list)
            s['name'] = series_name
            s['is_summary'] = False
        else:
            step_list = series_list[series_name]['step']
            value_list = series_list[series_name]['value']

            for i in range(len(step_list)):
                step_list[i] = step_list[i][-1:]
                value_list[i] = value_list[i][-1:]

            s = _dist_series_merge(step_list, value_list)
            s['name'] = series_name
            s['is_summary'] = True

        filtered_track_data.append(s)

    return filtered_track_data


def get_merged_dist_metrics_tracking(run_uuid: str, get_all_data: bool) -> JSONResponse:
    r: Optional['run.Run'] = run.get(run_uuid)

    rank_uuids = []
    if r is not None:
        rank_uuids = r.get_rank_uuids()

    if r is None:
        response = JSONResponse({'series': {}, 'insights': []}, status_code=404)
        response.status_code = 404
    else:
        metric_list = [metrics.MetricsAnalysis(m) if m else None for m in metrics.mget(list(rank_uuids.values()))]
        metric_list = [m for m in metric_list if m is not None]
        track_data_list = [m.get_tracking() for m in metric_list]

        preference_data = []
        preferences_key = DistMetricsPreferencesIndex.get(run_uuid)
        mp: DistMetricsPreferencesModel
        if preferences_key:
            mp = preferences_key.load()
            preference_data = mp.get_data()['series_preferences']
        else:
            mp = DistMetricsPreferencesModel()
            mp.save()
            DistMetricsPreferencesIndex.set(run_uuid, mp.key)

        # update preferences incase it doesn't match with the series
        series_list_set = set()
        series_list = []
        for track_data in track_data_list:
            for track_item in track_data:
                if track_item['name'] not in series_list:
                    series_list.append(track_item['name'])
                    series_list_set.add(track_item['name'])
        if len(preference_data) == 0:
            preference_data = get_default_series_preference(series_list)
            mp.update_preferences({'series_preferences': preference_data})
            mp.save()
        elif len(preference_data) != len(series_list):
            preference_data = fill_preferences(series_list, preference_data)
            mp.update_preferences({'series_preferences': preference_data})
            mp.save()

        preference_data = merge_preferences(run_uuid, preference_data)
        merged_tracking = get_merged_metric_tracking_util(track_data_list, preference_data, get_all_data)

        response = JSONResponse({'series': merged_tracking, 'insights': []})
        response.status_code = 200

    return response
