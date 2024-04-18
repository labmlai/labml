import {WeyaElement, WeyaElementFunction,} from '../../../../../lib/weya/weya'
import {AnalysisPreferenceModel} from "../../../models/preferences"
import {Card, CardOptions} from "../../types"
import {fillPlotPreferences} from "../../../components/charts/utils"
import {ROUTER} from '../../../app'
import {DataLoader} from '../../../components/loader'
import {CardWrapper} from "../chart_wrapper/card"
import metricsCache from "./cache"
import {CustomMetric, Indicator} from "../../../models/run"
import CACHE from "../../../cache/cache"


export class MetricsCard extends Card {
    private readonly uuid: string
    private readonly width: number
    private readonly metricUuid?: string
    private series: Indicator[]
    private preferenceData: AnalysisPreferenceModel
    private elem: HTMLDivElement
    private lineChartContainer: WeyaElement
    private loader: DataLoader
    private chartWrapper: CardWrapper
    private sparkLineContainer: WeyaElement
    private titleContainer: WeyaElement

    private customMetric?: CustomMetric

    constructor(opt: CardOptions) {
        super(opt)

        this.uuid = opt.uuid
        this.width = opt.width
        this.metricUuid = opt.params != null ? opt.params['custom_metric'] : null
        this.loader = new DataLoader(async (force) => {
            let analysisData = await  metricsCache.getAnalysis(this.uuid).get(force)
            this.series = analysisData.series

            if (this.metricUuid != null) {
                let customMetricList = await CACHE.getCustomMetrics(this.uuid).get(force)
                this.preferenceData = customMetricList.getMetric(this.metricUuid).preferences
                this.customMetric = customMetricList.getMetric(this.metricUuid)
            } else {
                this.preferenceData = await metricsCache.getPreferences(this.uuid).get(force)
            }

            this.preferenceData.series_preferences = fillPlotPreferences(this.series, this.preferenceData.series_preferences)
        })
    }

    getLastUpdated(): number {
        return metricsCache.getAnalysis(this.uuid).lastUpdated
    }

    async render($: WeyaElementFunction) {
        this.elem = $('div', '.labml-card.labml-card-action', {on: {click: this.onClick}}, $ => {
            this.titleContainer = $('h3','.header', 'Metrics')
            this.loader.render($)
            this.lineChartContainer = $('div', '')
            this.sparkLineContainer = $('div', '')
        })

        try {
            await this.loader.load()

            this.chartWrapper = new CardWrapper({
                elem: this.elem,
                preferenceData: this.preferenceData,
                series: this.series,
                lineChartContainer: this.lineChartContainer,
                sparkLinesContainer: this.sparkLineContainer,
                width: this.width
            })

            this.chartWrapper.render()
            this.renderDetails()
        } catch (e) {
        }
    }

    private renderDetails() {
        this.titleContainer.innerHTML = ''
        if (this.customMetric != null) {
            this.titleContainer.textContent = this.customMetric.name
        } else {
            this.titleContainer.textContent = 'Metrics'
        }
    }

    async refresh() {
        try {
            await this.loader.load(true)
            this.chartWrapper?.updateData(this.series, null, this.preferenceData)
            this.chartWrapper?.render()
        } catch (e) {
        }
    }

    onClick = () => {
        if (this.metricUuid != null) {
            ROUTER.navigate(`/run/${this.uuid}/metrics/${this.metricUuid}`)
        } else {
            ROUTER.navigate(`/run/${this.uuid}/metrics`)
        }
    }
}