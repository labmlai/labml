import {Weya as $, WeyaElement, WeyaElementFunction,} from '../../../../../lib/weya/weya'
import {InsightModel, SeriesModel} from "../../../models/run"
import {
    AnalysisPreferenceBaseModel,
    DistAnalysisPreferenceModel
} from "../../../models/preferences"
import {Card, CardOptions} from "../../types"
import CACHE from "../../../cache/cache"
import {getChartType, toPointValues} from "../../../components/charts/utils"
import {LineChart} from "../../../components/charts/lines/chart"
import {SparkLines} from "../../../components/charts/spark_lines/chart"
import InsightsList from "../../../components/insights_list"
import {ROUTER} from '../../../app'
import {DataLoader} from '../../../components/loader'
import {DistMetricsAnalysisCache, DistMetricsPreferenceCache} from "./cache"


export class DistributedMetricsCard extends Card {
    private readonly uuid: string
    private readonly width: number
    private series: SeriesModel[]
    private insights: InsightModel[]
    private preferenceData: DistAnalysisPreferenceModel
    private elem: HTMLDivElement
    private lineChartContainer: WeyaElement
    private insightsContainer: WeyaElement
    private loader: DataLoader
    private chartWrapper: MetricChartWrapper
    private sparkLineContainer: WeyaElement

    constructor(opt: CardOptions) {
        super(opt)

        this.uuid = opt.uuid
        this.width = opt.width

        this.loader = new DataLoader(async (force) => {
            let run = await CACHE.getRun(this.uuid).get(false)
            let worldSize = run.world_size

            if (worldSize == 0)
                return

            this.series = []
            this.insights = []
            let metricCache: DistMetricsAnalysisCache = new DistMetricsAnalysisCache(this.uuid, CACHE.getRunStatus(this.uuid))

            let analysisData = await metricCache.get(force)
            analysisData.series.forEach((series, index) => {
                let s: SeriesModel[] = toPointValues(series)
                for (let item of s) {
                    item.name = `rank${index+1} ${item.name}`
                }
                this.series = this.series.concat(s)
            })
            this.insights = analysisData.insights

            let preferenceCache = new DistMetricsPreferenceCache(this.uuid, worldSize, analysisData.series[0].length)
            this.preferenceData = await preferenceCache.get(force)
            console.log(this.preferenceData)
        })
    }

    getLastUpdated(): number {
        // todo implement this
        return 0
    }

    async render($: WeyaElementFunction) {
        this.elem = $('div', '.labml-card.labml-card-action', {on: {click: this.onClick}}, $ => {
            $('h3','.header', 'Distributed Metrics')
            this.loader.render($)
            this.lineChartContainer = $('div', '')
            this.sparkLineContainer = $('div', '')
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
                sparkLinesContainer: this.sparkLineContainer,
                width: this.width,
                isDistributed: true
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
        ROUTER.navigate(`/run/${this.uuid}/distributed`)
    }
}

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