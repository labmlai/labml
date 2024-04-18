import time
import uuid
from typing import List, Tuple, Any
from uuid import UUID

from labml_db import Model, Key, Index
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer
from fastapi import Request

from labml_app.analyses.analysis import Analysis
from labml_app.analyses.experiments.metrics import MetricsPreferencesModel


@Analysis.db_model(PickleSerializer, 'custom_metrics')
class CustomMetricModel(Model['CustomMetricModel']):
    preference_key: Key
    name: str
    description: str
    created_time: float
    metric_id: str

    @classmethod
    def defaults(cls):
        return dict(preference_key=None,
                    name='',
                    description='',
                    created_time=0,
                    metric_id=''
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
        mp = self.preference_key.load()
        return {
            'id': self.metric_id,
            'name': self.name,
            'description': self.description,
            'preferences': mp.get_data(),
            'created_time': self.created_time,
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

        mp = MetricsPreferencesModel()
        mp.save()
        cm.preference_key = mp.key

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
        return [k[1].load().get_data() for k in self.metrics]

    def update(self, data: dict):
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
    else:
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
