from typing import Any, Dict, List
from ..series import Series


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


def get_merged_metric_tracking_util(track_data_list, indicators: List[str]):
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
    for series_name in series_names:
        include_full_data = series_name in indicators

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
