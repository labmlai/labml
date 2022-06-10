from typing import Dict, Any, List, Union, Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from labml_db import Model, Index
from labml_db.serializer.pickle import PickleSerializer

from labml_app.enums import SeriesEnums
from labml_app.logger import logger
from .. import preferences
from ..analysis import Analysis
from ..series import Series
from ..series import SeriesModel
from ..series_collection import SeriesCollection
from ...db import user


class HyperParamPreferences(preferences.Preferences):
    sub_series_preferences: Dict[str, preferences.SeriesPreferences]

    @classmethod
    def defaults(cls):
        return dict(sub_series_preferences={},
                    )

    def update_sub_series_preferences(self, data: preferences.PreferencesData) -> None:
        data = data.get('sub_series_preferences', {})
        for k, v in data.items():
            self.sub_series_preferences[k] = v

        self.save()

    def get_sub_series_preferences(self) -> Dict[str, preferences.SeriesPreferences]:
        res = {}
        for k, v in self.sub_series_preferences.items():
            if v:
                res[k] = v
            else:
                res[k] = []

        return res

    def get_data(self) -> Dict[str, Any]:
        return {
            'series_preferences': self.series_preferences,
            'chart_type': self.chart_type,
            'sub_series_preferences': self.get_sub_series_preferences(),
        }


@Analysis.db_model(PickleSerializer, 'hyperparams')
class HyperParamsModel(Model['HyperParamsModel'], SeriesCollection):
    default_values: Dict[str, any]  # hp configs
    hp_values: Dict[str, Union[int, float]]  # current hp values
    hp_series: Dict[str, SeriesModel]  # values and steps updates
    has_hp_updated: Dict[str, bool]

    @classmethod
    def defaults(cls):
        return dict(
            default_values={},
            hp_values={},
            hp_series={},
            has_hp_updated={},
        )


@Analysis.db_model(PickleSerializer, 'hyperparams_preferences')
class HyperParamsPreferencesModel(Model['HyperParamsPreferencesModel'], HyperParamPreferences):
    pass


@Analysis.db_index(PickleSerializer, 'hyperparams_preferences')
class HyperParamsPreferencesIndex(Index['HyperParamsPreferences']):
    pass


@Analysis.db_index(PickleSerializer, 'hyperparams_index')
class HyperParamsIndex(Index['HyperParams']):
    pass


class HyperParamsAnalysis(Analysis):
    hyper_params: HyperParamsModel

    def __init__(self, data):
        self.hyper_params = data

    def track(self, data: Dict[str, SeriesModel]):
        res = {}
        for ind, s in data.items():
            ind_type = ind.split('.')[0]
            if ind_type == SeriesEnums.HYPERPARAMS:
                res[ind] = s

        self.hyper_params.track(res)

    def get_tracking(self):
        res = []
        default_values = self.hyper_params.default_values
        for ind, track in self.hyper_params.tracking.items():
            name_split = ind.split('.')
            name = ''.join(name_split[-1])

            s = Series().load(track)
            step, value = s.step.tolist(), s.value.tolist()
            series: Dict[str, Any] = {'step': step,
                                      'value': value,
                                      'smoothed': value,
                                      'is_editable': name in default_values,
                                      'range': default_values.get(name, {'range': []})['range'],
                                      'dynamic_type': default_values.get(name, {'dynamic_type': ''})[
                                          'dynamic_type'],
                                      'name': name}

            if name in self.hyper_params.hp_series:
                s = self.hyper_params.hp_series[name]
                steps, values = self.get_input_series(step[0],
                                                      s['step'],
                                                      s['value'],
                                                      self.hyper_params.step,
                                                      default_values[name]['default'])

                series['sub'] = {'step': steps, 'value': values, 'smoothed': values}

            res.append(series)

        res.sort(key=lambda s: s['name'])

        return res

    @staticmethod
    def get_input_series(start_step: int,
                         series_steps: List[float],
                         series_values: List[float],
                         current_step,
                         default: float):
        steps, values = [start_step, series_steps[0] - 1], [default, default]

        for i in range(len(series_steps)):
            v = series_values[i]

            if i + 1 > len(series_steps) - 1:
                ns = current_step - 1
            else:
                ns = series_steps[i + 1] - 1

            steps += [series_steps[i], ns]
            values += [v, v]

        if values:
            steps.append(current_step)
            values.append(values[-1])

        return steps, values

    @staticmethod
    def get_or_create(run_uuid: str):
        hyper_params_key = HyperParamsIndex.get(run_uuid)

        if not hyper_params_key:
            hp = HyperParamsModel()
            hp.save()
            HyperParamsIndex.set(run_uuid, hp.key)

            hpp = HyperParamsPreferencesModel()
            hpp.save()
            HyperParamsPreferencesIndex.set(run_uuid, hpp.key)

            return HyperParamsAnalysis(hp)

        return HyperParamsAnalysis(hyper_params_key.load())

    @staticmethod
    def delete(run_uuid: str):
        hyper_params_key = HyperParamsIndex.get(run_uuid)
        preferences_key = HyperParamsPreferencesIndex.get(run_uuid)

        if hyper_params_key:
            hp: HyperParamsModel = hyper_params_key.load()
            HyperParamsIndex.delete(run_uuid)
            hp.delete()

        if preferences_key:
            gp: HyperParamsPreferencesModel = preferences_key.load()
            HyperParamsPreferencesIndex.delete(run_uuid)
            gp.delete()

    def set_hyper_params(self, data: Dict[str, any]) -> None:
        hp_values = self.hyper_params.hp_values

        for k, v in data.items():
            if k not in hp_values:
                continue

            try:
                new_value = float(v)
                current_value = hp_values[k]

                if current_value and current_value == new_value:
                    continue

                hp_values[k] = new_value
                self.update_hp_series(k, new_value)
                self.hyper_params.has_hp_updated[k] = True
            except ValueError:
                logger.error(f'not a number : {v}')

        self.hyper_params.save()

    def update_hp_series(self, ind: str, value: float) -> None:
        hp_series = self.hyper_params.hp_series

        if ind not in hp_series:
            hp_series[ind] = {'step': [], 'value': []}

        hp_series[ind]['value'].append(value)
        hp_series[ind]['step'].append(self.hyper_params.step)

    def set_default_values(self, data: Dict[str, any]):
        default_values = {}
        hp_values = {}

        for k, v in data.items():
            default_values[k] = v
            hp_values[k] = v['default']

        self.hyper_params.default_values = default_values
        self.hyper_params.hp_values = hp_values
        self.hyper_params.save()

    def get_hyper_params(self):
        res = {}
        has_hp_updated = self.hyper_params.has_hp_updated
        for k, v in self.hyper_params.hp_values.items():
            if has_hp_updated.get(k, False):
                res[k] = v
                has_hp_updated[k] = False

                self.hyper_params.save()

        return res


@Analysis.route('GET', 'hyper_params/{run_uuid}')
async def get_hyper_params_tracking(request: Request, run_uuid: str) -> Any:
    track_data = []
    status_code = 404

    ans = HyperParamsAnalysis.get_or_create(run_uuid)
    if ans:
        track_data = ans.get_tracking()
        status_code = 200

    response = JSONResponse({'series': track_data, 'insights': []})
    response.status_code = status_code

    return response


@Analysis.route('GET', 'hyper_params/preferences/{run_uuid}')
async def get_hyper_params_preferences(request: Request, run_uuid: str) -> Any:
    preferences_data = {}

    preferences_key = HyperParamsPreferencesIndex.get(run_uuid)
    if not preferences_key:
        return preferences_data

    hpp: HyperParamsPreferencesModel = preferences_key.load()

    return hpp.get_data()


@Analysis.route('POST', 'hyper_params/preferences/{run_uuid}')
async def set_hyper_params_preferences(request: Request, run_uuid: str) -> Any:
    preferences_key = HyperParamsPreferencesIndex.get(run_uuid)

    if not preferences_key:
        return {}

    hpp = preferences_key.load()
    json = await request.json()
    hpp.update_preferences(json)

    logger.debug(f'update hyper_params preferences: {hpp.key}')

    return {'errors': hpp.errors}


@Analysis.route('POST', 'hyper_params/{run_uuid}', True)
async def set_hyper_params(request: Request, run_uuid: str, token: Optional[str] = None) -> Any:
    p = user.get_by_session_token(token).default_project
    if not p.is_project_run(run_uuid):
        return {'errors': []}

    ans = HyperParamsAnalysis.get_or_create(run_uuid)
    if ans:
        json = await request.json()
        ans.set_hyper_params(json)

    return {'errors': []}
