from enum import Enum
from typing import Dict, List, Any, Union

from labml_db import Model, Index
from labml_db.serializer.pickle import PickleSerializer
from labml_db.serializer.yaml import YamlSerializer

from labml_app.analyses.analysis import Analysis
from labml_app.db import run

PreferencesData = Dict[str, Any]
SeriesPreferences = List[int]


class SmoothFunction(Enum):
    Exponential = 'exponential'
    LeftExponential = 'left_exponential'


class Preferences:
    series_preferences: Union[SeriesPreferences, List['SeriesPreferences']]
    # series_preferences content:
    # -1 -> do not send data init
    # 0 -> send data init but not selected
    # 1 -> send data init and selected
    chart_type: int
    errors: List[Dict[str, str]]
    step_range: List[int]
    focus_smoothed: bool
    smooth_value: float
    smooth_function: str
    base_series_preferences: SeriesPreferences
    base_experiment: str
    is_base_distributed: bool

    @classmethod
    def defaults(cls):
        return dict(series_preferences=[],
                    chart_type=0,
                    errors=[],
                    step_range=[-1, -1],
                    focus_smoothed=True,
                    smooth_value=0.5,  # 50% smooth
                    smooth_function=SmoothFunction.LeftExponential.value,
                    is_base_distributed=False,
                    base_series_preferences=[],
                    base_experiment = "",
                    )

    def update_preferences(self, data: PreferencesData) -> None:
        if 'series_preferences' in data:
            self.update_series_preferences(data['series_preferences'])

        if 'chart_type' in data:
            self.chart_type = data['chart_type']

        if 'step_range' in data:
            self.step_range = data['step_range']

        if 'focus_smoothed' in data:
            self.focus_smoothed = data['focus_smoothed']

        if 'smooth_value' in data:
            self.smooth_value = data['smooth_value']

        if 'smooth_function' in data:
            self.smooth_function = data['smooth_function']

        if 'base_experiment' in data:
            self.base_experiment = data['base_experiment']

        if 'base_series_preferences' in data:
            self.base_series_preferences = data['base_series_preferences']

        r = run.get(self.base_experiment)
        if r is not None and r.world_size > 0:  # distributed run
            self.is_base_distributed = True
        else:
            self.is_base_distributed = False

        self.save()

    def update_series_preferences(self, data: SeriesPreferences) -> None:
        self.series_preferences = data

    def update_base_series_preferences(self, data: SeriesPreferences) -> None:
        self.base_series_preferences = data

    def get_data(self) -> Dict[str, Any]:
        return {
            'base_series_preferences': self.base_series_preferences,
            'series_preferences': self.series_preferences,
            'base_experiment': self.base_experiment,
            'chart_type': self.chart_type,
            'step_range': self.step_range,
            'focus_smoothed': self.focus_smoothed,
            'is_base_distributed': self.is_base_distributed,
            'smooth_value': self.smooth_value,
            'smooth_function': self.smooth_function,
        }


class MetricPreferences(Preferences):
    series_preferences: List[str]
    base_series_preferences: List[str]


@Analysis.db_model(PickleSerializer, 'metric_preferences')
class MetricPreferenceModel(Model['MetricPreferencesModel'], Preferences):
    pass


@Analysis.db_index(YamlSerializer, 'metric_preferences_index.yaml')
class MetricPreferenceIndex(Index['MetricPreferences']):
    pass
