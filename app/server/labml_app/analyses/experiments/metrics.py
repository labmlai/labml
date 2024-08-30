from typing import Dict, Any, List, Optional

from labml_db import Model, Index, load_keys
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer

from labml_app.enums import INDICATORS
from ..analysis import Analysis
from ..series import SeriesModel, Series
from ..series_collection import SeriesCollection
from labml_app.settings import INDICATOR_LIMIT


@Analysis.db_model(PickleSerializer, 'metrics')
class MetricsModel(Model['MetricsModel'], SeriesCollection):
    pass


@Analysis.db_index(YamlSerializer, 'metrics_index.yaml')
class MetricsIndex(Index['Metrics']):
    pass


def mget(run_uuids: List[str]) -> List[Optional['MetricsAnalysis']]:
    run_keys = MetricsIndex.mget(run_uuids)
    return load_keys(run_keys)


class MetricsAnalysis(Analysis):
    metrics: MetricsModel

    def __init__(self, data):
        self.metrics = data

    def track(self, data: Dict[str, SeriesModel], run_uuid: str = None) -> int:
        res = {}
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

        return self.metrics.track(res)

    def get_tracking(self):
        res = []
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

        if metrics_key:
            m: MetricsModel = metrics_key.load()
            MetricsIndex.delete(run_uuid)
            m.delete()


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
