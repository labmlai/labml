from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from labml_db import Model, Index
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer

from ..analysis import Analysis
from . import metrics
from ..preferences import Preferences
from ...logger import logger


@Analysis.db_model(PickleSerializer, 'dist_metrics_preferences')
class DistMetricsPreferencesModel(Model['DistMetricsPreferencesModel'], Preferences):
    pass


@Analysis.db_index(YamlSerializer, 'dist_metrics_preferences_index.yaml')
class DistMetricsPreferencesIndex(Index['DistMetricsPreferences']):
    pass


@Analysis.route('GET', 'distributed/metrics/preferences/{run_uuid}')
async def get_metrics_preferences(request: Request, run_uuid: str) -> Any:
    preferences_key = DistMetricsPreferencesIndex.get(run_uuid)
    if not preferences_key:
        mp = DistMetricsPreferencesModel()
        mp.save()
        preferences_key = mp.key

    mp: DistMetricsPreferencesModel = preferences_key.load()

    return mp.get_data()


@Analysis.route('POST', 'distributed/metrics/preferences/{run_uuid}')
async def set_metrics_preferences(request: Request, run_uuid: str) -> Any:
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

    ans = metrics.MetricsAnalysis.get_or_create(run_uuid)
    if ans:
        track_data = ans.get_tracking()
        status_code = 200

    response = JSONResponse({'series': track_data, 'insights': []})
    response.status_code = status_code

    return response
