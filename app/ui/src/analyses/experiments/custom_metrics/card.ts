import {Card, CardOptions} from "../../types"
import {Weya as $, WeyaElementFunction} from "../../../../../lib/weya/weya"
import CACHE, {AnalysisDataCache} from "../../../cache/cache"
import metricsCache from "./cache"
import {DataLoader} from "../../../components/loader"
import {ComparisonPreferenceModel} from "../../../models/preferences"
import {DEBUG} from "../../../env"
import {fillPlotPreferences} from "../../../components/charts/utils"
import {AnalysisData, CustomMetric, Indicator} from "../../../models/run"
import {ROUTER} from "../../../app"
import {NetworkError} from "../../../network"
import {CardWrapper} from "../chart_wrapper/card"

export class MetricCard extends Card {
    private readonly  currentUUID: string
    private baseUUID: string
    private readonly width: number
    private baseAnalysisCache: AnalysisDataCache
    private baseSeries: Indicator[]
    private currentSeries: Indicator[]
    private currentAnalysisCache: AnalysisDataCache
    private preferenceData: ComparisonPreferenceModel
    private loader: DataLoader
    private lineChartContainer: HTMLDivElement
    private sparkLineContainer: HTMLDivElement
    private elem: HTMLDivElement
    private chartWrapper: CardWrapper
    private customMetricUUID: string
    private titleContainer: HTMLElement
    private customMetric: CustomMetric

    constructor(opt: CardOptions) {
        super(opt)

        this.currentUUID = opt.uuid
        this.width = opt.width
        this.currentAnalysisCache = metricsCache.getAnalysis(this.currentUUID)
        this.currentAnalysisCache.setCurrentUUID(this.currentUUID)

        this.customMetricUUID = opt.params != null ? opt.params['custom_metric'] : null

        this.loader = new DataLoader(async (force: boolean) => {
            let customMetricList = await CACHE.getCustomMetrics(this.currentUUID).get(force)
            if (customMetricList == null || customMetricList.getMetric(this.customMetricUUID) == null) {
                throw new NetworkError(404, "", "Custom metric list is null")
            }

            this.customMetric = customMetricList.getMetric(this.customMetricUUID)
            this.preferenceData = customMetricList.getMetric(this.customMetricUUID).preferences
            this.baseUUID = this.preferenceData.base_experiment

            let currentAnalysisData: AnalysisData = await this.currentAnalysisCache.get(force)
            this.currentSeries = currentAnalysisData.series
            this.preferenceData.series_preferences = fillPlotPreferences(this.currentSeries, this.preferenceData.series_preferences)

            if (!!this.baseUUID) {
                this.baseAnalysisCache = metricsCache.getAnalysis(this.baseUUID)
                this.baseAnalysisCache.setCurrentUUID(this.currentUUID)
                try {
                    let baseAnalysisData = await this.baseAnalysisCache.get(force)
                    this.baseSeries = baseAnalysisData.series
                    this.preferenceData.base_series_preferences = fillPlotPreferences(this.baseSeries, this.preferenceData.base_series_preferences)
                } catch (e) {
                    if (e instanceof NetworkError && e.statusCode === 404) {
                    } else {
                        throw e
                    }
                }
            }
        })

    }

    getLastUpdated(): number {
        return this.currentAnalysisCache.lastUpdated
    }

    async refresh() {
        try {
             await this.loader.load(true)
             if (this.currentSeries.concat(this.baseSeries).length > 0) {
                this.chartWrapper?.updateData(this.currentSeries, this.baseSeries, this.preferenceData)
                this.chartWrapper?.render()
             }
         } catch (e) {
         }
    }

    async render($: WeyaElementFunction) {
        this.elem = $('div', '.labml-card.labml-card-action', {on: {click: this.onClick}}, $ => {
            this.titleContainer = $('h3', '.header', 'Metrics')
            this.loader.render($)

            this.lineChartContainer = $('div', '')
            this.sparkLineContainer = $('div', '')
        })

        try {
            await this.loader.load()

            this.chartWrapper = new CardWrapper({
                elem: this.elem,
                preferenceData: this.preferenceData,
                series: this.currentSeries,
                baseSeries: this.baseSeries,
                lineChartContainer: this.lineChartContainer,
                sparkLinesContainer: this.sparkLineContainer,
                width: this.width
            })

            this.chartWrapper.render()
            this.renderDetails()
        } catch (e) {
            if (DEBUG) {
                console.error(e)
            }
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

    onClick = () => {
       ROUTER.navigate(`/run/${this.currentUUID}/metrics/${this.customMetricUUID}`)
    }
}