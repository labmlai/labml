from typing import Dict, Any

from fastapi import Request
from fastapi.responses import JSONResponse
from labml_db import Model, Index
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer

from labml_app.logger import logger
from labml_app.enums import SeriesEnums
from labml_app.settings import INDICATOR_LIMIT
from ..analysis import Analysis
from ..series import SeriesModel, Series
from ..series_collection import SeriesCollection
from ..preferences import Preferences
from .. import helper


@Analysis.db_model(PickleSerializer, 'outputs')
class OutputsModel(Model['OutputsModel'], SeriesCollection):
    pass


@Analysis.db_model(PickleSerializer, 'outputs_preferences')
class OutputsPreferencesModel(Model['OutputsPreferencesModel'], Preferences):
    pass


@Analysis.db_index(YamlSerializer, 'outputs_preferences_index.yaml')
class OutputsPreferencesIndex(Index['OutputsPreferences']):
    pass


@Analysis.db_index(YamlSerializer, 'outputs_index.yaml')
class OutputsIndex(Index['Outputs']):
    pass


class OutputsAnalysis(Analysis):
    outputs: OutputsModel

    def __init__(self, data):
        self.outputs = data

    def track(self, data: Dict[str, SeriesModel]):
        res = {}
        for ind, s in data.items():
            ind_split = ind.split('.')
            ind_type = ind_split[0]
            ind_name = '.'.join(ind_split[:-1])
            if ind_type == SeriesEnums.MODULE:
                if ind_name not in self.outputs.indicators:
                    if len(self.outputs.indicators) >= INDICATOR_LIMIT:
                        continue
                    self.outputs.indicators.add(ind_name)

                res[ind] = s

        self.outputs.track(res)

    def get_track_summaries(self):
        data = {}
        for ind, track in self.outputs.tracking.items():
            name_split = ind.split('.')
            ind = name_split[-1]
            name = '.'.join(name_split[1:-1])

            series: Dict[str, Any] = Series().load(track).summary

            if name in data:
                data[name][ind] = series['mean']
            else:
                data[name] = {ind: series['mean']}

        if not data:
            return []

        sort_key = 'var'

        res = [v for k, v in data.items() if sort_key in v]
        sorted_res = sorted(res, key=lambda k: k[sort_key])

        ret = {}
        for d in sorted_res:
            for k, v in d.items():
                if k not in ret:
                    ret[k] = {'name': k, 'value': []}
                else:
                    ret[k]['value'].append(v)

        return [v for k, v in ret.items()]

    def get_tracking(self):
        res = []
        is_series_updated = False
        for ind, track in self.outputs.tracking.items():
            name = ind.split('.')
            if name[-1] != 'var':
                continue
            name = name[1:-1]

            s = Series().load(track)
            series: Dict[str, Any] = s.detail
            series['name'] = '.'.join(name)

            if s.is_smoothed_updated:
                self.outputs.tracking[ind] = s.to_data()
                is_series_updated = True

            res.append(series)

        if is_series_updated:
            self.outputs.save()

        res.sort(key=lambda s: s['mean'], reverse=True)

        helper.remove_common_prefix(res, 'name')

        return res

    @staticmethod
    def get_or_create(run_uuid: str):
        outputs_key = OutputsIndex.get(run_uuid)

        if not outputs_key:
            o = OutputsModel()
            o.save()
            OutputsIndex.set(run_uuid, o.key)

            op = OutputsPreferencesModel()
            op.save()
            OutputsPreferencesIndex.set(run_uuid, op.key)

            return OutputsAnalysis(o)

        return OutputsAnalysis(outputs_key.load())

    @staticmethod
    def delete(run_uuid: str):
        outputs_key = OutputsIndex.get(run_uuid)
        preferences_key = OutputsPreferencesIndex.get(run_uuid)

        if outputs_key:
            o: OutputsModel = outputs_key.load()
            OutputsIndex.delete(run_uuid)
            o.delete()

        if preferences_key:
            op: OutputsPreferencesModel = preferences_key.load()
            OutputsPreferencesIndex.delete(run_uuid)
            op.delete()


# @utils.mix_panel.MixPanelEvent.time_this(None)
@Analysis.route('GET', 'outputs/{run_uuid}')
async def get_modules_tracking(request: Request, run_uuid: str) -> Any:
    track_data = []
    summary_data = []
    status_code = 404

    ans = OutputsAnalysis.get_or_create(run_uuid)
    if ans:
        track_data = ans.get_tracking()
        summary_data = ans.get_track_summaries()
        status_code = 200

    response = JSONResponse({'series': track_data, 'insights': [], 'summary': summary_data})
    response.status_code = status_code

    return response


@Analysis.route('GET', 'outputs/preferences/{run_uuid}')
async def get_modules_preferences(request: Request, run_uuid: str) -> Any:
    preferences_data = {}

    preferences_key = OutputsPreferencesIndex.get(run_uuid)
    if not preferences_key:
        return preferences_data

    op: OutputsPreferencesModel = preferences_key.load()

    return op.get_data()


@Analysis.route('POST', 'outputs/preferences/{run_uuid}')
async def set_modules_preferences(request: Request, run_uuid: str) -> Any:
    preferences_key = OutputsPreferencesIndex.get(run_uuid)

    if not preferences_key:
        return {}

    op = preferences_key.load()
    json = await request.json()
    op.update_preferences(json)

    logger.debug(f'update outputs preferences: {op.key}')

    return {'errors': op.errors}
