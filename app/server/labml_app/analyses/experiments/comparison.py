from typing import Any, Dict

from fastapi import Request
from labml_db import Model, Index
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer

from labml_app.logger import logger
from ..analysis import Analysis
from .. import preferences


class ComparisonPreferences(preferences.Preferences):
    base_series_preferences: preferences.SeriesPreferences
    base_experiment: str

    @classmethod
    def defaults(cls):
        return dict(base_series_preferences=[],
                    base_experiment=str,
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

        self.save()

    def update_base_series_preferences(self, data: preferences.SeriesPreferences) -> None:
        self.base_series_preferences = data

    def get_data(self) -> Dict[str, Any]:
        return {
            'base_series_preferences': self.base_series_preferences,
            'series_preferences': self.series_preferences,
            'base_experiment': self.base_experiment,
            'chart_type': self.chart_type,
        }


@Analysis.db_model(PickleSerializer, 'comparison_preferences')
class ComparisonPreferencesModel(Model['ComparisonPreferencesModel'], ComparisonPreferences):
    pass


@Analysis.db_index(YamlSerializer, 'comparison_preferences_index.yaml')
class ComparisonPreferencesIndex(Index['ComparisonPreferences']):
    pass


@Analysis.route('GET', 'compare/preferences/{run_uuid}')
async def get_comparison_preferences(request: Request, run_uuid: str) -> Any:
    preferences_data = {}

    preferences_key = ComparisonPreferencesIndex.get(run_uuid)
    if not preferences_key:
        return preferences_data

    cp: ComparisonPreferences = preferences_key.load()
    preferences_data = cp.get_data()

    return preferences_data


@Analysis.route('POST', 'compare/preferences/{run_uuid}')
async def set_comparison_preferences(request: Request, run_uuid: str) -> Any:
    preferences_key = ComparisonPreferencesIndex.get(run_uuid)

    if not preferences_key:
        cp = ComparisonPreferencesModel()
        ComparisonPreferencesIndex.set(run_uuid, cp.key)
    else:
        cp = preferences_key.load()

    json = await request.json()
    cp.update_preferences(json)

    logger.debug(f'update comparison preferences: {cp.key}')

    return {'errors': cp.errors}
