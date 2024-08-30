from typing import Any

from fastapi import Request
from starlette.responses import JSONResponse

from labml_app import utils
from .custom_metrics import CustomMetricsListIndex, CustomMetricsListModel
from .distributed_metrics import get_merged_metric_tracking_util
from .metrics import MetricsAnalysis, get_metrics_tracking_util, mget
from ..analysis import Analysis
from ...db import run
from ...utils import merge_preferences


@Analysis.route('POST', 'compare/metrics/{run_uuid}')
async def get_comparison_metrics(request: Request, run_uuid: str) -> Any:
    get_all_data = (await request.json())['get_all']
    current_uuid = request.query_params['current']
    metric_uuid = request.query_params['metric']
    is_base = current_uuid != run_uuid

    r = run.get(run_uuid)
    if r is None:
        response = JSONResponse({'series': [], 'insights': []})
        response.status_code = 200
        return response

    list_key = CustomMetricsListIndex.get(run_uuid)

    if list_key is None:
        response = JSONResponse({'error': 'Custom Metric list for the run not found'})
        response.status_code = 404
        return response

    custom_metric_list: CustomMetricsListModel = list_key.load()

    matching_metrics = [m for m in custom_metric_list.metrics if m[0] == metric_uuid]
    if len(matching_metrics) == 0:
        response = JSONResponse({'error': 'Custom Metric not found'})
        response.status_code = 404
        return response

    custom_metric = matching_metrics[0][1].load()
    preference_data = custom_metric.get_data()['preferences']

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
            custom_metric.update({
                'preferences': {'base_series_preferences' if is_base else 'series_preferences': preference_data}})
        elif len(preference_data) != len(track_data):
            preference_data = utils.fill_preferences([s['name'] for s in track_data], preference_data)
            custom_metric.update({
                'preferences': {'base_series_preferences' if is_base else 'series_preferences': preference_data}})
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
            custom_metric.update({
                'preferences': {'base_series_preferences' if is_base else 'series_preferences': preference_data}})
        elif len(preference_data) != len(series_list):
            preference_data = utils.fill_preferences(series_list, preference_data)
            custom_metric.update({
                'preferences': {'base_series_preferences' if is_base else 'series_preferences': preference_data}})

        preference_data = merge_preferences(run_uuid, preference_data)
        merged_tracking = get_merged_metric_tracking_util(track_data_list, preference_data, get_all_data)

        response = JSONResponse({'series': merged_tracking, 'insights': []})
        response.status_code = 200
        return response
