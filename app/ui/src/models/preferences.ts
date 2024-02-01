
export interface AnalysisPreferenceBaseModel {
    series_preferences: number[] | number[][]
    sub_series_preferences: Object
    chart_type: number
    step_range: number[]
    focus_smoothed: boolean
}

export interface AnalysisPreferenceModel extends AnalysisPreferenceBaseModel {
    series_preferences: number[]
}

export interface ComparisonPreferenceModel extends AnalysisPreferenceModel {
    base_series_preferences: number[]
    base_experiment: string
    is_base_distributed: boolean
}