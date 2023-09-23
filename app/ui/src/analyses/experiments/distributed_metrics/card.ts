import {Weya as $, WeyaElement, WeyaElementFunction,} from '../../../../../lib/weya/weya'
import {InsightModel, SeriesModel} from "../../../models/run"
import {AnalysisPreferenceModel} from "../../../models/preferences"
import {Card, CardOptions} from "../../types"
import {AnalysisDataCache, AnalysisPreferenceCache} from "../../../cache/cache"
import {getChartType, toPointValues} from "../../../components/charts/utils"
import {LineChart} from "../../../components/charts/lines/chart"
import metricsCache from "../metrics/cache"
import {SparkLines} from "../../../components/charts/spark_lines/chart"
import InsightsList from "../../../components/insights_list"
import {ROUTER} from '../../../app'
import {DataLoader} from '../../../components/loader'

export class MetricsCard extends Card {
    uuid: string
    width: number
    series: SeriesModel[]
    insights: InsightModel[]
    preferenceData: AnalysisPreferenceModel
    analysisCache: AnalysisDataCache
    elem: HTMLDivElement
    lineChartContainer: WeyaElement
    sparkLinesContainer: WeyaElement
    insightsContainer: WeyaElement
    preferenceCache: AnalysisPreferenceCache
    plotIdx: number[] = []
    private loader: DataLoader
    private chartWrapper: MetricChartWrapper

    constructor(opt: CardOptions) {
        super(opt)

        this.uuid = opt.uuid
        this.width = opt.width
        this.analysisCache = metricsCache.getAnalysis(this.uuid)
        this.preferenceCache = metricsCache.getPreferences(this.uuid)
        this.loader = new DataLoader(async (force) => {
            let analysisData = await this.analysisCache.get(force)
            this.series = toPointValues(analysisData.series)
            this.insights = analysisData.insights
            this.preferenceData = await this.preferenceCache.get(force)
        })
    }

    getLastUpdated(): number {
        return this.analysisCache.lastUpdated
    }

    async render($: WeyaElementFunction) {
        this.elem = $('div', '.labml-card.labml-card-action', {on: {click: this.onClick}}, $ => {
            $('h3','.header', 'Metrics')
            this.loader.render($)
            this.lineChartContainer = $('div', '')
            this.sparkLinesContainer = $('div', '')
            this.insightsContainer = $('div', '')
        })

        try {
            await this.loader.load()

            this.chartWrapper = new MetricChartWrapper({
                elem: this.elem,
                preferenceData: this.preferenceData,
                insights: this.insights,
                series: this.series,
                insightsContainer: this.insightsContainer,
                lineChartContainer: this.lineChartContainer,
                sparkLinesContainer: this.sparkLinesContainer,
                width: this.width
            })

            this.chartWrapper.render()
        } catch (e) {
        }
    }

    async refresh() {
        try {
            await this.loader.load(true)
            this.chartWrapper?.updateData(this.series, this.insights, this.preferenceData)
            this.chartWrapper?.render()
        } catch (e) {
        }
    }

    onClick = () => {
        ROUTER.navigate(`/run/${this.uuid}/metrics`)
    }
}

interface MetricChartWrapperOptions {
    width: number
    series: SeriesModel[]
    insights: InsightModel[]

    lineChartContainer: WeyaElement
    sparkLinesContainer: WeyaElement
    insightsContainer: WeyaElement
    elem: WeyaElement

    preferenceData: AnalysisPreferenceModel
}

class MetricChartWrapper {
    private width: number
    private series: SeriesModel[]
    private insights: InsightModel[]

    private readonly lineChartContainer: WeyaElement
    private readonly sparkLinesContainer: WeyaElement
    private readonly insightsContainer: WeyaElement
    private readonly elem: WeyaElement

    private plotIdx: number[] = []
    private chartType: number
    private stepRange: number[]
    private focusSmoothed: boolean

    constructor(opt: MetricChartWrapperOptions) {
        this.elem = opt.elem
        this.width = opt.width
        this.lineChartContainer = opt.lineChartContainer
        this.sparkLinesContainer = opt.sparkLinesContainer
        this.insightsContainer = opt.insightsContainer

        this.updateData(opt.series, opt.insights, opt.preferenceData)
    }

    public updateData(series: SeriesModel[], insights: InsightModel[],preferenceData: AnalysisPreferenceModel) {
        this.series = series
        this.insights = insights

        let analysisPreferences = preferenceData.series_preferences
        if (analysisPreferences.length > 0) {
            this.plotIdx = [...analysisPreferences]
        } else {
            this.plotIdx = []
        }

        this.chartType = preferenceData.chart_type
        this.stepRange = preferenceData.step_range
        this.focusSmoothed = preferenceData.focus_smoothed
    }

    public render() {
        console.log(this.lineChartContainer)
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
        this.lineChartContainer.innerHTML = ''
        $(this.lineChartContainer, $ => {
            new LineChart({
                series: this.series,
                width: this.width,
                plotIdx: this.plotIdx,
                chartType: this.chartType != null ? getChartType(this.chartType) : 'linear',
                isDivergent: true,
                stepRange: this.stepRange,
                focusSmoothed: this.focusSmoothed
            }).render($)
        })
    }

    private renderSparkLines() {
        this.sparkLinesContainer.innerHTML = ''
        $(this.sparkLinesContainer, $ => {
            new SparkLines({
                series: this.series,
                plotIdx: this.plotIdx,
                width: this.width,
                isDivergent: true
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