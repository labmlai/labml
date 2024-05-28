
export interface AnalysisPreferenceBaseModel {
    series_preferences: number[] | number[][]
    sub_series_preferences: Object
    chart_type: number
    step_range: number[]
    focus_smoothed: boolean
    smooth_value: number
    smooth_function: string
}

export interface AnalysisPreferenceModel extends AnalysisPreferenceBaseModel {
    series_preferences: number[]
    series_names?: string[]
}

export interface ComparisonPreferenceModel extends AnalysisPreferenceModel {
    base_series_preferences: number[]
    base_experiment: string
    is_base_distributed: boolean
    base_series_names?: string[]
}