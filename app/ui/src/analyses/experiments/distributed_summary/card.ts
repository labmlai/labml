import {WeyaElement, WeyaElementFunction, Weya as $} from '../../../../../lib/weya/weya'
import {InsightModel, SeriesModel} from "../../../models/run"
import {
    AnalysisPreferenceModel,
} from "../../../models/preferences"
import {Card, CardOptions} from "../../types"
import CACHE, {AnalysisPreferenceCache} from "../../../cache/cache"
import {toPointValues} from "../../../components/charts/utils"
import {DataLoader} from '../../../components/loader'
import {DistMetricsAnalysisCache} from "../distributed_metrics/cache"
import {MetricChartWrapper} from "../distributed_metrics/card"


export class MetricsSummaryCard extends Card {
    private readonly uuid: string
    private readonly width: number

    private worldSize: number

    private series: SeriesModel[][]
    private preferenceData: AnalysisPreferenceModel
    private elem: HTMLDivElement
    private lineChartContainer: WeyaElement
    private lineChartItems: WeyaElement[]
    private loader: DataLoader
    private chartWrapper: MetricChartWrapper[]
    private sparkLineContainer: WeyaElement
    private insightsContainer: WeyaElement
    private insights: InsightModel[]

    constructor(opt: CardOptions) {
        super(opt)

        this.uuid = opt.uuid
        this.width = opt.width

        this.chartWrapper = []

        this.loader = new DataLoader(async (force) => {
            let run = await CACHE.getRun(this.uuid).get(false)
            this.worldSize = run.world_size
            if (this.worldSize == 0)
                return

            this.series = []
            let metricCache: DistMetricsAnalysisCache = new DistMetricsAnalysisCache(this.uuid, CACHE.getRunStatus(this.uuid))

            let analysisData = await metricCache.get(force)
            for (let series of analysisData.series) {
                this.series.push(toPointValues(series))
            }
            this.insights = analysisData.insights

            let preferenceCache = new AnalysisPreferenceCache(this.uuid, 'metrics')
            this.preferenceData = await preferenceCache.get(force)
        })
    }

    getLastUpdated(): number {
        // todo implement this
        return 0
    }

    private initLineCharts() {
        this.lineChartItems = []
        $(this.lineChartContainer, $ => {
            for (let i=0; i<this.worldSize; i++) {
                let item = $('div.fit-content')
                this.lineChartItems.push(item)
            }
        })
    }

    async render($: WeyaElementFunction) {
        this.elem = $('div', '.labml-card.labml-card-action', {on: {click: this.onClick}}, $ => {
            $('h3','.header', 'Metrics Summary')
            this.loader.render($)

            this.lineChartContainer = $('div.metric-summary-container')
            this.sparkLineContainer = $('div', '')
            this.insightsContainer = $('div', '')
        })

        try {
            await this.loader.load()
            this.initLineCharts()
            for (let i=0; i<this.worldSize; i++) {
                let chart = new MetricChartWrapper({
                    elem: this.elem,
                    preferenceData: this.preferenceData,
                    insights: this.insights,
                    series: this.series[i],
                    insightsContainer: this.insightsContainer,
                    lineChartContainer: this.lineChartItems[i],
                    sparkLinesContainer: this.sparkLineContainer,
                    width: this.width/2-50,
                    isDistributed: false
                })
                this.chartWrapper.push(chart)
                chart.render()
            }
        } catch (e) {
        }
    }

    async refresh() {
        try {
            await this.loader.load(true)
            this.chartWrapper?.forEach((chart, index) => {
                chart.updateData(this.series[index], this.insights, this.preferenceData)
                chart.render()
            })

        } catch (e) {
        }
    }

    onClick = () => {
        // ROUTER.navigate(`/run/${this.uuid}/distributed`)
    }
}