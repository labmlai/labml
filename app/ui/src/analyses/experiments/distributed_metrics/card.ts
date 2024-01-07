import {Weya as $, WeyaElement} from '../../../../../lib/weya/weya'
import {InsightModel, SeriesModel} from "../../../models/run"
import {
    AnalysisPreferenceBaseModel,
} from "../../../models/preferences"
import {getChartType} from "../../../components/charts/utils"
import {LineChart} from "../../../components/charts/lines/chart"
import {SparkLines} from "../../../components/charts/spark_lines/chart"
import InsightsList from "../../../components/insights_list"

interface MetricChartWrapperOptions {
    width: number
    series: SeriesModel[]
    insights: InsightModel[]
    isDistributed: boolean

    lineChartContainer: WeyaElement
    sparkLinesContainer?: WeyaElement
    insightsContainer: WeyaElement
    elem: WeyaElement

    preferenceData: AnalysisPreferenceBaseModel

    title?: string
    showValues?: boolean
}

export class MetricChartWrapper {
    private width: number
    private series: SeriesModel[]
    private insights: InsightModel[]
    private isDistributed: boolean

    private readonly lineChartContainer: WeyaElement
    private readonly sparkLinesContainer?: WeyaElement
    private readonly insightsContainer: WeyaElement
    private readonly elem: WeyaElement

    private plotIdx: number[] = []
    private chartType: number
    private stepRange: number[]
    private focusSmoothed: boolean

    private readonly title?: string
    private readonly showValues?: boolean

    constructor(opt: MetricChartWrapperOptions) {
        this.elem = opt.elem
        this.width = opt.width
        this.isDistributed = opt.isDistributed
        this.lineChartContainer = opt.lineChartContainer
        this.sparkLinesContainer = opt.sparkLinesContainer
        this.insightsContainer = opt.insightsContainer
        this.title = opt.title
        this.showValues = opt.showValues ?? true

        this.updateData(opt.series, opt.insights, opt.preferenceData)
    }

    public updateData(series: SeriesModel[], insights: InsightModel[],preferenceData: AnalysisPreferenceBaseModel) {
        this.series = series
        this.insights = insights

        let analysisPreferences = preferenceData.series_preferences
        if (analysisPreferences.length > 0) {
            this.plotIdx = [].concat(...analysisPreferences)
        } else {
            this.plotIdx = []
        }

        this.chartType = preferenceData.chart_type
        this.stepRange = preferenceData.step_range
        this.focusSmoothed = preferenceData.focus_smoothed
    }

    public render() {
        if (this.series.length > 0) {
            this.elem.classList.remove('hide')
            this.renderLineChart()
            this.renderSparkLines()
            this.renderInsights()
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
                width: this.width,
                plotIdx: this.plotIdx,
                chartType: this.chartType != null ? getChartType(this.chartType) : 'linear',
                isDivergent: true,
                stepRange: this.stepRange,
                focusSmoothed: this.focusSmoothed,
                isDistributed: this.isDistributed
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
                plotIdx: this.plotIdx,
                width: this.width,
                isDivergent: true,
                isDistributed: this.isDistributed,
                onlySelected: true,
                showValue: this.showValues
            }).render($)
        })
    }

    private renderInsights() {
        this.insightsContainer.innerHTML = ''
        $(this.insightsContainer, $ => {
            new InsightsList({insightList: this.insights}).render($)
        })
    }
}