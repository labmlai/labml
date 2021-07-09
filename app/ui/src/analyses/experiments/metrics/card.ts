import {Weya as $, WeyaElement, WeyaElementFunction,} from '../../../../../lib/weya/weya'
import {InsightModel, SeriesModel} from "../../../models/run"
import {AnalysisPreferenceModel} from "../../../models/preferences"
import {Card, CardOptions} from "../../types"
import {AnalysisDataCache, AnalysisPreferenceCache} from "../../../cache/cache"
import {getChartType, toPointValues} from "../../../components/charts/utils"
import {LineChart} from "../../../components/charts/lines/chart"
import metricsCache from "./cache"
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

            let analysisPreferences = this.preferenceData.series_preferences
            if (analysisPreferences.length > 0) {
                this.plotIdx = [...analysisPreferences]
            }

            if (this.series.length > 0) {
                this.renderLineChart()
                this.renderSparkLines()
                this.renderInsights()
            } else {
                this.elem.classList.add('hide')
            }
        } catch (e) {
        }
    }

    renderLineChart() {
        this.lineChartContainer.innerHTML = ''
        $(this.lineChartContainer, $ => {
            new LineChart({
                series: this.series,
                width: this.width,
                plotIdx: this.plotIdx,
                chartType: this.preferenceData && this.preferenceData.chart_type ?
                    getChartType(this.preferenceData.chart_type) : 'linear',
                isDivergent: true
            }).render($)
        })
    }

    renderSparkLines() {
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

    renderInsights() {
        this.insightsContainer.innerHTML = ''
        $(this.insightsContainer, $ => {
            new InsightsList({insightList: this.insights}).render($)
        })
    }

    async refresh() {
        try {
            await this.loader.load(true)
            if (this.series.length > 0) {
                this.renderLineChart()
                this.renderSparkLines()
                this.renderInsights()
                this.elem.classList.remove('hide')
            }
        } catch (e) {
        }
    }

    onClick = () => {
        ROUTER.navigate(`/run/${this.uuid}/metrics`)
    }
}
