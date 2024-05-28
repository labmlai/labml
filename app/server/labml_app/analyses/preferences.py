from enum import Enum
from typing import Dict, List, Any, Union

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

    @classmethod
    def defaults(cls):
        return dict(series_preferences=[],
                    chart_type=0,
                    errors=[],
                    step_range=[-1, -1],
                    focus_smoothed=True,
                    smooth_value=0.5,  # 50% smooth
                    smooth_function=SmoothFunction.Exponential.value
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

        self.save()

    def update_series_preferences(self, data: SeriesPreferences) -> None:
        self.series_preferences = data

    def get_data(self) -> Dict[str, Any]:
        return {
            'series_preferences': self.series_preferences,
            'chart_type': self.chart_type,
            'step_range': self.step_range,
            'focus_smoothed': self.focus_smoothed,
            'smooth_value': self.smooth_value,
            'smooth_function': self.smooth_function,
        }
