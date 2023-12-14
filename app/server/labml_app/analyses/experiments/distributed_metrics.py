from typing import Any, Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from labml_db import Model, Index
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer

from ..analysis import Analysis
from . import metrics
from ..preferences import Preferences
from ..series import Series
from ...logger import logger
from ...db import run


@Analysis.db_model(PickleSerializer, 'merged_metrics_preferences')
class DistMetricsPreferencesModel(Model['DistMetricsPreferencesModel'], Preferences):
    pass


@Analysis.db_index(YamlSerializer, 'merged_metrics_preferences_index.yaml')
class DistMetricsPreferencesIndex(Index['DistMetricsPreferences']):
    pass


@Analysis.route('GET', 'distributed/metrics/merged/preferences/{run_uuid}')
async def get_merged_metrics_preferences(request: Request, run_uuid: str) -> Any:
    preferences_key = DistMetricsPreferencesIndex.get(run_uuid)
    if not preferences_key:
        mp = DistMetricsPreferencesModel()
        mp.save()
        preferences_key = mp.key

    mp: DistMetricsPreferencesModel = preferences_key.load()

    return mp.get_data()


@Analysis.route('POST', 'distributed/metrics/merged/preferences/{run_uuid}')
async def set_merged_metrics_preferences(request: Request, run_uuid: str) -> Any:
    preferences_key = DistMetricsPreferencesIndex.get(run_uuid)

    mp = None
    if not preferences_key:
        mp = DistMetricsPreferencesModel()
        mp.save()
        DistMetricsPreferencesIndex.set(run_uuid, mp.key)

    if not mp:
        mp = preferences_key.load()
    json = await request.json()
    mp.update_preferences(json)

    logger.debug(f'update distributed metrics preferences: {mp.key}')

    return {'errors': mp.errors}


@Analysis.route('GET', 'distributed/metrics/merged/{run_uuid}')
async def get_merged_dist_metrics_tracking(request: Request, run_uuid: str) -> Any:
    track_data = []
    status_code = 404

    r: Optional['run.Run'] = run.get(run_uuid)

    if r is None:
        return JSONResponse({'series': {}, 'insights': []}, status_code=404)

    rank_uuids = r.get_other_uuids()

    if len(rank_uuids.keys()) == 0:  # not distributed main rank
        ans = metrics.MetricsAnalysis.get_or_create(run_uuid)
        if ans:
            track_data = ans.get_tracking()
            status_code = 200

        response = JSONResponse({'series': track_data, 'insights': []})
        response.status_code = status_code
    else:
        metric_list = [metrics.MetricsAnalysis(m) if m else None for m in metrics.mget(list(rank_uuids.values()))]
        metric_list = [m for m in metric_list if m is not None]
        track_data_list = [m.get_tracking() for m in metric_list]
        series_list = {}
        for track_data in track_data_list:
            for track_item in track_data:
                if track_item['name'] not in series_list:
                    series_list[track_item['name']] = track_item
                series_list[track_item['name']]['step'] += track_item['step']
                series_list[track_item['name']]['value'] += track_item['value']

        merged_list = []

        for key in series_list:
            steps, values = zip(*sorted(zip(series_list[key]['step'], series_list[key]['value'])))
            series_list[key]['step'] = list(steps)
            series_list[key]['value'] = list(values)

            s = Series().load(Series().to_data())
            s.update(series_list[key]['step'], series_list[key]['value'])
            details = s.detail
            details['name'] = key
            merged_list.append(details)

        response = JSONResponse({'series': merged_list, 'insights': []})
        response.status_code = 200

    return response
