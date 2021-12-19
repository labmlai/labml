from typing import Dict, Any

from fastapi import Request
from fastapi.responses import JSONResponse
from labml_db import Model, Index
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer

from labml_app.logger import logger
from labml_app import enums
from labml_app import settings
from ..analysis import Analysis
from ..series import SeriesModel, Series
from ..series_collection import SeriesCollection
from ..preferences import Preferences
from .. import helper


@Analysis.db_model(PickleSerializer, 'gradients')
class GradientsModel(Model['GradientsModel'], SeriesCollection):
    pass


@Analysis.db_model(PickleSerializer, 'gradients_preferences')
class GradientsPreferencesModel(Model['GradientsPreferencesModel'], Preferences):
    pass


@Analysis.db_index(YamlSerializer, 'gradients_preferences_index.yaml')
class GradientsPreferencesIndex(Index['GradientsPreferences']):
    pass


@Analysis.db_index(YamlSerializer, 'gradients_index.yaml')
class GradientsIndex(Index['Gradients']):
    pass


class GradientsAnalysis(Analysis):
    gradients: GradientsModel

    def __init__(self, data):
        self.gradients = data

    def track(self, data: Dict[str, SeriesModel]):
        res: Dict[str, SeriesModel] = {}
        for ind, s in data.items():
            ind_split = ind.split('.')
            ind_type = ind_split[0]
            ind_name = '.'.join(ind_split[:-1])
            if ind_type == enums.SeriesEnums.GRAD:
                if ind_name not in self.gradients.indicators:
                    if len(self.gradients.indicators) >= settings.INDICATOR_LIMIT:
                        continue
                    self.gradients.indicators.add(ind_name)

                res[ind] = s

        self.gradients.track(res)

    def get_tracking(self):
        res = []
        is_series_updated = False
        for ind, track in self.gradients.tracking.items():
            name = ind.split('.')
            if name[-1] != 'l2':
                continue
            name = name[1:-1]

            s = Series().load(track)
            series: Dict[str, Any] = s.detail
            series['name'] = '.'.join(name)

            if s.is_smoothed_updated:
                self.gradients.tracking[ind] = s.to_data()
                is_series_updated = True

            res.append(series)

        if is_series_updated:
            self.gradients.save()

        res.sort(key=lambda s: s['mean'], reverse=True)

        helper.remove_common_prefix(res, 'name')

        return res

    def get_track_summaries(self):
        data = {}
        for ind, track in self.gradients.tracking.items():
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

        sort_key = 'l2'

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

    @staticmethod
    def get_or_create(run_uuid: str):
        gradients_key = GradientsIndex.get(run_uuid)

        if not gradients_key:
            g = GradientsModel()
            g.save()
            GradientsIndex.set(run_uuid, g.key)

            gp = GradientsPreferencesModel()
            gp.save()
            GradientsPreferencesIndex.set(run_uuid, gp.key)

            return GradientsAnalysis(g)

        return GradientsAnalysis(gradients_key.load())

    @staticmethod
    def delete(run_uuid: str):
        gradients_key = GradientsIndex.get(run_uuid)
        preferences_key = GradientsPreferencesIndex.get(run_uuid)

        if gradients_key:
            g: GradientsModel = gradients_key.load()
            GradientsIndex.delete(run_uuid)
            g.delete()

        if preferences_key:
            gp: GradientsPreferencesModel = preferences_key.load()
            GradientsPreferencesIndex.delete(run_uuid)
            gp.delete()


@Analysis.route('GET', 'gradients/{run_uuid}')
async def get_grads_tracking(request: Request, run_uuid: str) -> Any:
    track_data = []
    summary_data = []
    status_code = 404

    ans = GradientsAnalysis.get_or_create(run_uuid)
    if ans:
        track_data = ans.get_tracking()
        summary_data = ans.get_track_summaries()
        status_code = 200

    response = JSONResponse({'series': track_data, 'insights': [], 'summary': summary_data})
    response.status_code = status_code

    return response


@Analysis.route('GET', 'gradients/preferences/{run_uuid}')
async def get_grads_preferences(request: Request, run_uuid: str) -> Any:
    preferences_data = {}

    preferences_key = GradientsPreferencesIndex.get(run_uuid)
    if not preferences_key:
        return preferences_data

    gp: GradientsPreferencesModel = preferences_key.load()

    return gp.get_data()


@Analysis.route('POST', 'gradients/preferences/{run_uuid}')
async def set_grads_preferences(request: Request, run_uuid: str) -> Any:
    preferences_key = GradientsPreferencesIndex.get(run_uuid)

    if not preferences_key:
        return {}

    gp = preferences_key.load()
    json = await request.json()
    gp.update_preferences(json)

    logger.debug(f'update gradients preferences: {gp.key}')

    return {'errors': gp.errors}
