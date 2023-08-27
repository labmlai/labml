export interface AnalysisPreferenceModel {
    series_preferences: number[]
    sub_series_preferences: Object
    chart_type: number
}

export class AnalysisPreference {
    series_preferences: number[]
    sub_series_preferences: object
    chart_type: number

    constructor(preference: AnalysisPreferenceModel) {
        if (preference.series_preferences) {
            this.series_preferences = preference.series_preferences
        } else {
            this.series_preferences = []
        }
        this.chart_type = preference.chart_type
        this.sub_series_preferences = preference.sub_series_preferences
    }
}

export interface ComparisonPreferenceModel extends AnalysisPreferenceModel {
    base_series_preferences: number[]
    base_experiment: string
    focus_current: boolean
}

export class ComparisonPreference extends AnalysisPreference {
    base_series_preferences: number[]
    compared_with: string

    constructor(preference: ComparisonPreferenceModel) {
        super(preference)
        if (preference.base_series_preferences) {
            this.base_series_preferences = preference.base_series_preferences
        } else {
            this.base_series_preferences = []
        }
        this.compared_with = preference.base_experiment
    }
}
