import {Weya as $, WeyaElement} from '../../../../../lib/weya/weya'
import {Indicator} from "../../../models/run"
import {
    AnalysisPreferenceModel, ComparisonPreferenceModel,
} from "../../../models/preferences"
import {getChartType, smoothAndTrimAllCharts} from "../../../components/charts/utils"
import {LineChart} from "../../../components/charts/lines/chart"
import {SparkLines} from "../../../components/charts/spark_lines/chart"

interface CardWrapperOptions {
    width: number
    series: Indicator[]
    baseSeries?: Indicator[]

    lineChartContainer: WeyaElement
    sparkLinesContainer?: WeyaElement
    elem: WeyaElement

    preferenceData: AnalysisPreferenceModel | ComparisonPreferenceModel

    title?: string
}

export class CardWrapper {
    private width: number
    private series: Indicator[]
    private baseSeries: Indicator[]

    private readonly lineChartContainer: WeyaElement
    private readonly sparkLinesContainer?: WeyaElement
    private readonly elem: WeyaElement

    private plotIdx: number[] = []
    private basePlotIdx: number[] = []
    private chartType: number
    private stepRange: number[]
    private focusSmoothed: boolean
    private smoothValue: number
    private trimSmoothEnds: boolean

    private readonly title?: string

    constructor(opt: CardWrapperOptions) {
        this.elem = opt.elem
        this.width = opt.width
        this.lineChartContainer = opt.lineChartContainer
        this.sparkLinesContainer = opt.sparkLinesContainer
        this.title = opt.title

        this.updateData(opt.series, opt.baseSeries,  opt.preferenceData)
    }

    public updateData(series: Indicator[], baseSeries: Indicator[], preferenceData: AnalysisPreferenceModel | ComparisonPreferenceModel) {
        this.series = series
        this.baseSeries = baseSeries ?? []

        let analysisPreferences = preferenceData.series_preferences
        if (analysisPreferences.length > 0) {
            this.plotIdx = [].concat(...analysisPreferences)
        } else {
            this.plotIdx = []
        }

        if ((<ComparisonPreferenceModel>preferenceData)?.base_series_preferences != null) {
            let baseAnalysisPreferences = (<ComparisonPreferenceModel>preferenceData).base_series_preferences
            if (baseAnalysisPreferences.length > 0) {
                this.basePlotIdx = [].concat(...baseAnalysisPreferences)
            } else {
                this.basePlotIdx = []
            }
        }

        this.chartType = preferenceData.chart_type
        this.stepRange = preferenceData.step_range
        this.focusSmoothed = preferenceData.focus_smoothed
        this.smoothValue = preferenceData.smooth_value
        this.trimSmoothEnds = preferenceData.trim_smooth_ends

        smoothAndTrimAllCharts(this.series, this.baseSeries, this.smoothValue, this.stepRange, this.trimSmoothEnds)
    }

    public render() {
        if (this.series.length + this.baseSeries.length > 0) {
            this.elem.classList.remove('hide')
            this.renderLineChart()
            this.renderSparkLines()
        } else {
            this.elem.classList.add('hide')
        }
    }

    private renderLineChart() {
        if (this.lineChartContainer == null) {
            return
        }
        this.lineChartContainer.innerHTML = ''
        $(this.lineChartContainer, $ => {
            if (this.title != null) {
                $('span', '.title.text-secondary', this.title)
            }
            new LineChart({
                series: this.series,
                baseSeries: this.baseSeries,
                width: this.width,
                plotIndex: this.plotIdx,
                basePlotIdx: this.basePlotIdx,
                chartType: this.chartType != null ? getChartType(this.chartType) : 'linear',
                isDivergent: true,
                stepRange: this.stepRange,
                focusSmoothed: this.focusSmoothed,
                smoothValue: this.smoothValue
            }).render($)
        })
    }

    private renderSparkLines() {
        if (this.sparkLinesContainer == null) {
            return
        }
        this.sparkLinesContainer.innerHTML = ''
        $(this.sparkLinesContainer, $ => {
            new SparkLines({
                series: this.series,
                baseSeries: this.baseSeries,
                plotIdx: this.plotIdx,
                basePlotIdx: this.basePlotIdx,
                width: this.width,
                isDivergent: true,
                onlySelected: true,
                smoothValue: this.smoothValue,
                isMouseMoveOpt: false
            }).render($)
        })
    }
}