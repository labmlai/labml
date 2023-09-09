from typing import Dict, List, Any

PreferencesData = Dict[str, Any]
SeriesPreferences = List[int]


class Preferences:
    series_preferences: SeriesPreferences
    chart_type: int
    errors: List[Dict[str, str]]
    step_range: List[int]

    @classmethod
    def defaults(cls):
        return dict(series_preferences=[],
                    chart_type=0,
                    errors=[],
                    step_range=[-1, -1],
                    )

    def update_preferences(self, data: PreferencesData) -> None:
        if 'series_preferences' in data:
            self.update_series_preferences(data['series_preferences'])

        if 'chart_type' in data:
            self.chart_type = data['chart_type']

        if 'step_range' in data:
            self.step_range = data['step_range']

        self.save()

    def update_series_preferences(self, data: SeriesPreferences) -> None:
        self.series_preferences = data

    def get_data(self) -> Dict[str, Any]:
        return {
            'series_preferences': self.series_preferences,
            'chart_type': self.chart_type,
            'step_range': self.step_range,
        }
