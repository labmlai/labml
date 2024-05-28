from typing import Any, Dict, List

from fastapi import Request
from labml_db import Model, Index
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer
from starlette.responses import JSONResponse

from labml_app import utils
from labml_app.logger import logger
from .distributed_metrics import get_merged_dist_metrics_tracking, get_merged_metric_tracking_util
from .metrics import MetricsAnalysis, get_metrics_tracking_util, mget
from ..analysis import Analysis
from .. import preferences
from ...db import run
from ...utils import merge_preferences


class ComparisonPreferences(preferences.Preferences):
    base_series_preferences: preferences.SeriesPreferences
    base_experiment: str
    is_base_distributed: bool

    @classmethod
    def defaults(cls):
        return dict(base_series_preferences=[],
                    base_experiment="",
                    step_range=[-1, -1],
                    is_base_distributed=False,
                    smooth_value=0.5,
                    smooth_function=preferences.SmoothFunction.Exponential.value
                    )

    def update_preferences(self, data: preferences.PreferencesData) -> None:
        if 'base_series_preferences' in data:
            self.update_base_series_preferences(data['base_series_preferences'])

        if 'base_experiment' in data:
            self.base_experiment = data['base_experiment']

        if 'series_preferences' in data:
            self.update_series_preferences(data['series_preferences'])

        if 'chart_type' in data:
            self.chart_type = data['chart_type']

        if 'step_range' in data:
            self.step_range = data['step_range']

        if 'focus_smoothed' in data:
            self.focus_smoothed = data['focus_smoothed']

        if 'smooth_value' in data:
            self.smooth_value = data['smooth_value']

        if 'smooth_function' in data:
            self.smooth_function = data['smooth_function']

        r = run.get(self.base_experiment)
        if r is not None and r.world_size > 0:  # distributed run
            self.is_base_distributed = True
        else:
            self.is_base_distributed = False

        self.save()

    def update_base_series_preferences(self, data: preferences.SeriesPreferences) -> None:
        self.base_series_preferences = data

    def get_data(self) -> Dict[str, Any]:
        return {
            'base_series_preferences': self.base_series_preferences,
            'series_preferences': self.series_preferences,
            'base_experiment': self.base_experiment,
            'chart_type': self.chart_type,
            'step_range': self.step_range,
            'focus_smoothed': self.focus_smoothed,
            'is_base_distributed': self.is_base_distributed,
            'smooth_value': self.smooth_value,
            'smooth_function': self.smooth_function,
        }


@Analysis.db_model(PickleSerializer, 'comparison_preferences')
class ComparisonPreferencesModel(Model['ComparisonPreferencesModel'], ComparisonPreferences):
    pass


@Analysis.db_index(YamlSerializer, 'comparison_preferences_index.yaml')
class ComparisonPreferencesIndex(Index['ComparisonPreferences']):
    pass


@Analysis.route('POST', 'compare/metrics/{run_uuid}')
async def get_comparison_metrics(request: Request, run_uuid: str) -> Any:
    get_all_data = (await request.json())['get_all']
    current_uuid = request.query_params['current']
    is_base = current_uuid != run_uuid

    r = run.get(run_uuid)
    if r is None:
        response = JSONResponse({'series': [], 'insights': []})
        response.status_code = 200
        return response

    preferences_key = ComparisonPreferencesIndex.get(current_uuid)
    cp: ComparisonPreferencesModel
    if preferences_key is None:
        cp = ComparisonPreferencesModel()
        cp.save()
        ComparisonPreferencesIndex.set(current_uuid, cp.key)
        preference_data = cp.get_data()
    else:
        cp = preferences_key.load()
        preference_data = cp.get_data()

    preference_data = preference_data['base_series_preferences'] if is_base else preference_data['series_preferences']
    status_code = 404
    track_data = []

    if r.world_size == 0:
        ans = MetricsAnalysis.get_or_create(run_uuid)
        if ans:
            track_data = ans.get_tracking()
            status_code = 200
        # update preferences incase it doesn't match with the series
        if len(preference_data) == 0:
            preference_data = utils.get_default_series_preference([s['name'] for s in track_data])
            cp.update_preferences({'base_series_preferences' if is_base else 'series_preferences': preference_data})
            cp.save()
        elif len(preference_data) != len(track_data):
            preference_data = utils.fill_preferences([s['name'] for s in track_data], preference_data)
            cp.update_preferences({'base_series_preferences' if is_base else 'series_preferences': preference_data})
            cp.save()
        filtered_track_data = get_metrics_tracking_util(track_data, preference_data,
                                                        get_all_data)
        response = JSONResponse({'series': filtered_track_data, 'insights': []})
        response.status_code = status_code

        return response
    else:  # distributed run
        rank_uuids = r.get_rank_uuids()

        metric_list = [MetricsAnalysis(m) if m else None for m in mget(list(rank_uuids.values()))]
        metric_list = [m for m in metric_list if m is not None]
        track_data_list = [m.get_tracking() for m in metric_list]

        # update preferences incase it doesn't match with the series
        series_list_set = set()
        series_list = []
        for track_data in track_data_list:
            for track_item in track_data:
                if track_item['name'] not in series_list_set:
                    series_list.append(track_item['name'])
                    series_list_set.add(track_item['name'])
        if len(preference_data) == 0:
            preference_data = utils.get_default_series_preference(series_list)
            cp.update_preferences({'base_series_preferences' if is_base else 'series_preferences': preference_data})
            cp.save()
        elif len(preference_data) != len(series_list):
            preference_data = utils.fill_preferences(series_list, preference_data)
            cp.update_preferences({'base_series_preferences' if is_base else 'series_preferences': preference_data})
            cp.save()

        preference_data = merge_preferences(run_uuid, preference_data)
        merged_tracking = get_merged_metric_tracking_util(track_data_list, preference_data, get_all_data)

        response = JSONResponse({'series': merged_tracking, 'insights': []})
        response.status_code = 200
        return response


@Analysis.route('GET', 'compare/preferences/{run_uuid}')
async def get_comparison_preferences(request: Request, run_uuid: str) -> Any:
    preferences_key = ComparisonPreferencesIndex.get(run_uuid)

    if not preferences_key:
        cp = ComparisonPreferencesModel()
        cp.save()
        ComparisonPreferencesIndex.set(run_uuid, cp.key)
    else:
        cp = preferences_key.load()

    preferences_data = cp.get_data()

    response = JSONResponse(preferences_data)
    response.status_code = 200
    return response


@Analysis.route('POST', 'compare/preferences/{run_uuid}')
async def set_comparison_preferences(request: Request, run_uuid: str) -> Any:
    preferences_key = ComparisonPreferencesIndex.get(run_uuid)

    if not preferences_key:
        cp = ComparisonPreferencesModel()
        cp.save()
        ComparisonPreferencesIndex.set(run_uuid, cp.key)
    else:
        cp = preferences_key.load()

    json = await request.json()

    app_preferences = json['series_preferences']
    client_preferences = cp.get_data()['series_preferences']

    if len(app_preferences) != len(client_preferences):
        if 'series_names' not in json:
            raise ValueError('series_names not found in the request')
        series_names = [s['name'] for s in MetricsAnalysis.get_or_create(run_uuid).get_tracking()]
        json['series_preferences'] = (
            utils.update_series_preferences(app_preferences, json['series_names'], series_names))

    if 'base_experiment' in json and json['base_experiment'] != '':
        base_app_preferences = json['base_series_preferences']
        base_client_preferences = cp.get_data()['base_series_preferences']
        if len(base_app_preferences) != len(base_client_preferences):
            if 'base_series_names' not in json:
                raise ValueError('base_series_names not found in the request')
            series_names = [s['name'] for s in MetricsAnalysis.get_or_create(json['base_experiment']).get_tracking()]
            json['base_series_preferences'] = (
                utils.update_series_preferences(base_app_preferences, json['base_series_names'], series_names))

    cp.update_preferences(json)

    logger.debug(f'update comparison preferences: {cp.key}')

    return {'errors': cp.errors}
