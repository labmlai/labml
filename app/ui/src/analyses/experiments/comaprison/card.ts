import {Card, CardOptions} from "../../types"
import {Weya as $, WeyaElementFunction} from "../../../../../lib/weya/weya"
import {AnalysisDataCache, AnalysisPreferenceCache, ComparisonAnalysisPreferenceCache} from "../../../cache/cache"
import comparisonCache from "./cache"
import {DataLoader} from "../../../components/loader"
import {ComparisonPreferenceModel} from "../../../models/preferences"
import {DEBUG} from "../../../env"
import {clearChildElements} from "../../../utils/document"
import {fillPlotPreferences, toPointValues} from "../../../components/charts/utils"
import {SeriesModel} from "../../../models/run"
import {ROUTER} from "../../../app"
import {NetworkError} from "../../../network"
import {CardWrapper} from "../chart_wrapper/card"
import metricsCache from "./cache";

export class ComparisonCard extends Card {
    private readonly  currentUUID: string
    private baseUUID: string
    private readonly width: number
    private baseAnalysisCache: AnalysisDataCache
    private baseSeries: SeriesModel[]
    private currentSeries: SeriesModel[]
    private currentAnalysisCache: AnalysisDataCache
    private preferenceCache: ComparisonAnalysisPreferenceCache
    private preferenceData: ComparisonPreferenceModel
    private loader: DataLoader
    private missingBaseExperiment: boolean
    private lineChartContainer: HTMLDivElement
    private sparkLineContainer: HTMLDivElement
    private elem: HTMLDivElement
    private chartWrapper: CardWrapper

    constructor(opt: CardOptions) {
        super(opt)

        this.currentUUID = opt.uuid
        this.width = opt.width
        this.currentAnalysisCache = comparisonCache.getAnalysis(this.currentUUID)
        this.preferenceCache = <ComparisonAnalysisPreferenceCache>comparisonCache.getPreferences(this.currentUUID)
        this.currentAnalysisCache.setCurrentUUID(this.currentUUID)

        this.loader = new DataLoader(async (force: boolean) => {
            this.preferenceData = <ComparisonPreferenceModel> await this.preferenceCache.get(force)
            this.baseUUID = this.preferenceData.base_experiment

            let currentAnalysisData = await this.currentAnalysisCache.get(force)
            this.currentSeries = toPointValues(currentAnalysisData.series)
            this.preferenceData.series_preferences = fillPlotPreferences(this.currentSeries, this.preferenceData.series_preferences)

            if (!!this.baseUUID) {
                this.baseAnalysisCache = comparisonCache.getAnalysis(this.baseUUID)
                this.baseAnalysisCache.setCurrentUUID(this.currentUUID)
                try {
                    let baseAnalysisData = await this.baseAnalysisCache.get(force)
                    this.baseSeries = toPointValues(baseAnalysisData.series)
                    this.preferenceData.base_series_preferences = fillPlotPreferences(this.baseSeries, this.preferenceData.base_series_preferences)
                    this.missingBaseExperiment = false
                } catch (e) {
                    if (e instanceof NetworkError && e.statusCode === 404) {
                        this.missingBaseExperiment = true
                    } else {
                        throw e
                    }
                }
            } else {
                this.missingBaseExperiment = true
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
            $('h3', '.header', 'Comparison')
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

            if (!this.missingBaseExperiment) {
                this.chartWrapper.render()
            } else {
                this.renderEmptyChart()
            }
        } catch (e) {
            if (DEBUG) {
                console.error(e)
            }
        }
    }

    private renderEmptyChart() {
        clearChildElements(this.lineChartContainer)
        $(this.lineChartContainer, $ => {
            $('div', '.empty-chart-message', `${screen.width < 500 ? "Tap" : "Click"} here to compare with another experiment`)
        })
    }

    onClick = () => {
       ROUTER.navigate(`/run/${this.currentUUID}/compare`)
    }
}