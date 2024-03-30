import time
from typing import List, Tuple
from uuid import UUID

from labml_db import Model, Key, Index
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer
from requests import Request

from labml_app.analyses.analysis import Analysis
from labml_app.analyses.experiments.metrics import MetricsPreferencesModel


@Analysis.db_model(PickleSerializer, 'custom_metrics')
class CustomMetricModel(Model['CustomMetricModel']):
    preference_key: Key
    name: str
    description: str
    created_time: float
    metric_id: str

    def __init__(self, data: dict, **kwargs):
        super().__init__(**kwargs)
        self.name = data['name']
        self.description = data.get('description', '')
        self.metric_id = UUID().hex
        self.created_time = time.time()

        mp = MetricsPreferencesModel()
        mp.save()
        self.preference_key = mp.key

    def update_preferences(self, data: dict):
        mp = self.preference_key.load()
        mp.update_preferences(data)
        mp.save()

    def get_data(self):
        mp = self.preference_key.load()
        return {
            'id': self.metric_id,
            'name': self.name,
            'description': self.description,
            'preferences': mp.get_data()
        }


@Analysis.db_index(YamlSerializer, 'custom_metrics_list')
class CustomMetricsListIndex(Index['CustomMetricsListIndex']):
    pass


@Analysis.db_model(PickleSerializer, 'custom_metrics_list')
class CustomMetricsListModel(Model['CustomMetricsListModel']):
    metrics: List[Tuple[str, Key[CustomMetricModel]]]

    def __init__(self, run_uuid: str, **kwargs):
        super().__init__(**kwargs)
        self.metrics = []
        self.save()
        CustomMetricsListIndex.set(run_uuid, self.key)

    def create_custom_metric(self, data: dict):
        cm = CustomMetricModel(data)
        cm.save()
        self.metrics.append((cm.metric_id, cm.key))
        self.save()

        return cm.get_data()

    def delete_custom_metric(self, key: Key[CustomMetricModel]):
        self.metrics = [k for k in self.metrics if k != key]
        self.save()
        key.delete()

    def get_data(self):
        return [k[1].load().get_data() for k in self.metrics]

    def update(self, data: dict):
        for (metric_id, key) in self.metrics:
            if metric_id == data['id']:
                key.load().update_preferences(data)
                break


@Analysis.route('GET', 'custom_metrics/{run_uuid}')
def get_custom_metrics(request: Request, run_uuid: str):
    list_key = CustomMetricsListIndex.get(run_uuid)

    if list_key is None:
        r = CustomMetricsListModel(run_uuid)
    else:
        r = list_key.load()

    return {'metrics': r.get_data()}


@Analysis.route('POST', 'custom_metrics/{run_uuid}')
def update_custom_metric(request: Request, run_uuid: str):
    data = request.json()
    list_key = CustomMetricsListIndex.get(run_uuid)

    if list_key is None:
        r = CustomMetricsListModel(run_uuid)
    else:
        r = list_key.load()

    r.update(data)

    return {'status': 'success'}


@Analysis.route('POST', 'custom_metrics/{run_uuid}/create')
def create_custom_metric(request: Request, run_uuid: str):
    data = request.json()
    list_key = CustomMetricsListIndex.get(run_uuid)

    if list_key is None:
        r = CustomMetricsListModel(run_uuid)
    else:
        r = list_key.load()

    custom_metric_data = r.create_custom_metric(data)

    return custom_metric_data


@Analysis.route('POST', 'custom_metrics/{run_uuid}/delete')
def delete_custom_metric(request: Request, run_uuid: str):
    data = request.json()
    list_key = CustomMetricsListIndex.get(run_uuid)

    if list_key is None:
        return {'status': 'error', 'message': 'No custom metrics found'}

    r = list_key.load()
    r.delete_custom_metric(data['key'])

    return {'status': 'success'}
