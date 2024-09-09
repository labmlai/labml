import time
import uuid
from typing import List, Tuple, Any
from uuid import UUID

from labml_app.analyses.helper import get_similarity
from labml_app.db import user, run
from labml_db import Model, Key, Index
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer
from fastapi import Request

from labml_app.analyses.analysis import Analysis
from labml_app.analyses.experiments.metrics import MetricsAnalysis
from labml_app.analyses.preferences import MetricPreferenceModel


@Analysis.db_model(PickleSerializer, 'custom_metrics')
class CustomMetricModel(Model['CustomMetricModel']):
    preference_key: Key
    name: str
    description: str
    created_time: float
    metric_id: str
    position: int

    @classmethod
    def defaults(cls):
        return dict(preference_key=None,
                    name='',
                    description='',
                    created_time=0,
                    metric_id='',
                    position=1
                    )

    def update(self, data: dict):
        if 'preferences' in data:
            mp = self.preference_key.load()
            mp.update_preferences(data['preferences'])
            mp.save()

        if 'name' in data:
            self.name = data['name']
        if 'description' in data:
            self.description = data['description']
        self.save()

    def get_data(self):
        try:
            mp = self.preference_key.load()
        except KeyError:  # backward compatibility
            mp = MetricPreferenceModel()
            mp.save()
            self.preference_key = mp.key
            self.save()
        return {
            'id': self.metric_id,
            'name': self.name,
            'description': self.description,
            'preferences': mp.get_data(),
            'created_time': self.created_time,
            'position': self.position
        }


@Analysis.db_index(YamlSerializer, 'custom_metrics_list')
class CustomMetricsListIndex(Index['CustomMetricsListIndex']):
    pass


@Analysis.db_model(PickleSerializer, 'custom_metrics_list')
class CustomMetricsListModel(Model['CustomMetricsListModel']):
    metrics: List[Tuple[str, Key[CustomMetricModel]]]

    @classmethod
    def defaults(cls):
        return dict(metrics=[]
                    )

    def create_custom_metric(self, data: dict):
        cm = CustomMetricModel()
        cm.name = data['name']
        cm.description = data.get('description', '')
        cm.metric_id = uuid.uuid4().hex
        cm.created_time = time.time()
        cm.position = len(self.metrics) + 1

        mp = MetricPreferenceModel()
        mp.save()
        cm.preference_key = mp.key

        if 'preferences' in data:
            mp.update_preferences(data['preferences'])
            mp.save()

        cm.save()
        self.metrics.append((cm.metric_id, cm.key))
        self.save()

        return cm.get_data()

    def delete_custom_metric(self, metric_id: str):
        for (m_id, key) in self.metrics:
            if metric_id == m_id:
                key.delete()
                break
        self.metrics = [(m_id, key) for (m_id, key) in self.metrics if m_id != metric_id]
        self.save()

    def get_data(self):
        metrics = [k[1].load() for k in self.metrics]
        metrics = sorted(metrics, key=lambda x: (x.position, -1*x.created_time))
        for i, metric in enumerate(metrics, start=1):
            if metric.position != i:
                metric.position = i
                metric.save()
        return [m.get_data() for m in metrics]

    def get_metrics(self):
        return [k[1].load() for k in self.metrics]

    def update(self, data: dict):
        if 'position' in data:  # handle position separately
            metrics = [k[1].load() for k in self.metrics]
            metrics = sorted(metrics, key=lambda x: (x.position, -1*x.created_time))
            current_metric = [m for m in metrics if m.metric_id == data['id']]
            if len(current_metric) == 0:
                raise KeyError('Metric not found')
            current_metric = current_metric[0]
            metrics = [m for m in metrics if m.metric_id != data['id']]
            metrics.insert(data['position']-1, current_metric)
            for i, m in enumerate(metrics, start=1):
                m.position = i
                m.save()

        for (metric_id, key) in self.metrics:
            if metric_id == data['id']:
                key.load().update(data)
                break


@Analysis.route('GET', 'custom_metrics/{run_uuid}')
async def get_custom_metrics(request: Request, run_uuid: str) -> Any:
    list_key = CustomMetricsListIndex.get(run_uuid)

    if list_key is None:
        r = CustomMetricsListModel()
        r.save()
        CustomMetricsListIndex.set(run_uuid, r.key)
        await create_magic_metric(request, run_uuid)
        list_key = r.key

    r = list_key.load()
    return {'metrics': r.get_data()}


@Analysis.route('POST', 'custom_metrics/{run_uuid}')
async def update_custom_metric(request: Request, run_uuid: str) -> Any:
    data = await request.json()
    list_key = CustomMetricsListIndex.get(run_uuid)

    if list_key is None:
        r = CustomMetricsListModel()
        r.save()
        CustomMetricsListIndex.set(run_uuid, r.key)
    else:
        r = list_key.load()

    r.update(data)

    return {'status': 'success'}


@Analysis.route('POST', 'custom_metrics/{run_uuid}/create')
async def create_custom_metric(request: Request, run_uuid: str) -> Any:
    data = await request.json()
    list_key = CustomMetricsListIndex.get(run_uuid)

    if list_key is None:
        r = CustomMetricsListModel()
        r.save()
        CustomMetricsListIndex.set(run_uuid, r.key)
    else:
        r = list_key.load()

    custom_metric_data = r.create_custom_metric(data)

    return custom_metric_data


@Analysis.route('POST', 'custom_metrics/{run_uuid}/delete')
async def delete_custom_metric(request: Request, run_uuid: str) -> Any:
    data = await request.json()
    list_key = CustomMetricsListIndex.get(run_uuid)

    if list_key is None:
        return {'status': 'error', 'message': 'No custom metrics found'}

    r = list_key.load()
    r.delete_custom_metric(data['id'])

    return {'status': 'success'}


@Analysis.route('GET', 'custom_metrics/{run_uuid}/magic')
async def create_magic_metric(request: Request, run_uuid: str) -> Any:
    current_run = run.get(run_uuid)
    if current_run is None:
        return {'is_success': False, 'message': 'Run not found'}

    run_cm = CustomMetricsListIndex.get(run_uuid).load()

    current_run = current_run.get_summary()

    analysis_data = MetricsAnalysis.get_or_create(run_uuid).get_tracking()
    run_indicators = sorted([track_item['name'] for track_item in analysis_data])

    current_metrics = run_cm.get_data()
    current_created_charts = []
    current_selected_metrics = set()
    for x in current_metrics:
        current_created_charts.append(x['preferences']['series_preferences'])
        current_selected_metrics = current_selected_metrics.union(set(x['preferences']['series_preferences']))

    u = user.get_by_session_token('local')  # todo

    default_project = u.default_project
    runs = [r.get_summary() for r in default_project.get_runs()]

    runs = sorted(runs, key=lambda i: i['start_time'], reverse=True)
    similarity = [get_similarity(current_run, x) for x in runs]
    runs = [x for s, x in sorted(zip(similarity, runs), key=lambda pair: pair[0], reverse=True)]
    runs = runs[:20]

    indicator_counts = {}
    reasons = ""  # possible reason to not have runs
    for r in runs:
        list_key = CustomMetricsListIndex.get(r['run_uuid'])
        if list_key is None:
            continue
        cm = list_key.load()
        cm = cm.get_metrics()

        for m in cm:
            m_data = m.get_data()
            preferences = m_data['preferences']
            if len(preferences['series_preferences']) == 0:
                continue

            # has_same_chart = False
            # for indicator_list in current_created_charts:
            #     if sorted(indicator_list) ==
            #     sorted([x for x in preferences['series_preferences'] if x in run_indicators]):
            #         has_same_chart = True
            #         break
            # if has_same_chart:
            #     continue

            has_indicators = False
            for indicators in current_selected_metrics:
                if indicators in preferences['series_preferences']:
                    has_indicators = True
                    break

            if has_indicators:
                reasons = "Some charts ignored due to having overlapped indicators with current charts."
                continue

            preference_map_key = '|'.join(sorted(preferences['series_preferences']))
            if preference_map_key not in indicator_counts:
                indicator_counts[preference_map_key] = []
            indicator_counts[preference_map_key].append((m.key, m_data['created_time']))

    if len(indicator_counts) == 0:
        return {'is_success': False, 'message': "Couldn't find any new related chart. " + reasons}

    sorted_keys = sorted(indicator_counts.keys(), key=lambda x: len(indicator_counts[x]), reverse=True)

    # find the first indicator list with overlap
    selected = None
    for key in sorted_keys:
        ind = key.split('|')
        overlap = False
        for i in run_indicators:
            if i in ind:
                overlap = True
                break
        if overlap:
            selected = key
            break

    if selected is None:  # return smth
        return {'status': 'error', 'message': 'No similar indicators found from selected charts. ' + reasons}

    selected_metric = sorted(indicator_counts[selected], key=lambda x: x[1], reverse=True)[0]
    selected_metric = selected_metric[0].load()

    new_metric_data = selected_metric.get_data()
    new_metric_data['preferences']['series_preferences'] = \
        [x for x in new_metric_data['preferences']['series_preferences'] if x in run_indicators]

    run_cm.create_custom_metric(new_metric_data)

    return {'is_success': True}
