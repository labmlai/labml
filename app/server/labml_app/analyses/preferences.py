from typing import Dict, List, Any, Union

PreferencesData = Dict[str, Any]
SeriesPreferences = List[int]


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

    @classmethod
    def defaults(cls):
        return dict(series_preferences=[],
                    chart_type=0,
                    errors=[],
                    step_range=[-1, -1],
                    focus_smoothed=True,
                    smooth_value=50  # 50% smooth
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

        self.save()

    def update_series_preferences(self, data: SeriesPreferences) -> None:
        self.series_preferences = data

    def get_data(self) -> Dict[str, Any]:
        return {
            'series_preferences': self.series_preferences,
            'chart_type': self.chart_type,
            'step_range': self.step_range,
            'focus_smoothed': self.focus_smoothed,
            'smooth_value': self.smooth_value
        }
