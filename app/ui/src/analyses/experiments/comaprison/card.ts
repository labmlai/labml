import {Weya as $, WeyaElement, WeyaElementFunction,} from '../../../../../lib/weya/weya'
import {SeriesModel} from "../../../models/run"
import {ComparisonPreferenceModel} from "../../../models/preferences"
import {Card, CardOptions} from "../../types"
import {AnalysisDataCache, AnalysisPreferenceCache} from "../../../cache/cache"
import {getChartType, toPointValues} from "../../../components/charts/utils"
import {ROUTER} from '../../../app'
import {DataLoader} from '../../../components/loader'
import comparisonCache from './cache'
import {CompareLineChart} from '../../../components/charts/compare_lines/chart'
import {CompareSparkLines} from '../../../components/charts/compare_spark_lines/chart'
import {NetworkError} from '../../../network'
import {ErrorMessage} from '../../../components/error_message'
import {DEBUG} from '../../../env'

export class ComparisonCard extends Card {
    currentUuid: string
    baseUuid: string
    width: number
    currentSeries: SeriesModel[]
    baseSeries: SeriesModel[]
    preferenceData: ComparisonPreferenceModel
    baseAnalysisCache: AnalysisDataCache
    currentAnalysisCache: AnalysisDataCache
    elem: HTMLDivElement
    lineChartContainer: HTMLDivElement
    sparkLinesContainer: WeyaElement
    preferenceCache: AnalysisPreferenceCache
    currentPlotIdx: number[] = []
    basePlotIdx: number[] = []
    private loader: DataLoader
    private missingBaseExperiment: boolean

    constructor(opt: CardOptions) {
        super(opt)

        this.currentUuid = opt.uuid
        this.width = opt.width
        this.currentAnalysisCache = comparisonCache.getAnalysis(this.currentUuid)
        this.preferenceCache = comparisonCache.getPreferences(this.currentUuid)
        this.loader = new DataLoader(async (force) => {
            this.preferenceData = <ComparisonPreferenceModel>await this.preferenceCache.get(force)
            this.baseUuid = this.preferenceData.base_experiment

            let currentAnalysisData = await this.currentAnalysisCache.get(force)
            this.currentSeries = toPointValues(currentAnalysisData.series)
            if (this.baseUuid != null && this.baseUuid.length > 0) {
                this.baseAnalysisCache = comparisonCache.getAnalysis(this.baseUuid)
                try {
                    let baseAnalysisData = await this.baseAnalysisCache.get(force)
                    this.baseSeries = toPointValues(baseAnalysisData.series)
                    this.missingBaseExperiment = false
                } catch (e) {
                    if (e instanceof NetworkError && e.statusCode === 404) {
                        this.missingBaseExperiment = true
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

    async render($: WeyaElementFunction) {
        this.elem = $('div', '.labml-card.labml-card-action', {on: {click: this.onClick}}, $ => {
            $('h3', '.header', 'Comparison')
            this.loader.render($)
            this.lineChartContainer = $('div', '')
            this.sparkLinesContainer = $('div', '')
        })

        try {
            await this.loader.load()

            let currentAnalysisPreferences = this.preferenceData.series_preferences
            if (currentAnalysisPreferences.length > 0) {
                this.currentPlotIdx = [...currentAnalysisPreferences]
            }
            let baseAnalysisPreferences = this.preferenceData.base_series_preferences
            if (baseAnalysisPreferences.length > 0) {
                this.basePlotIdx = [...baseAnalysisPreferences]
            }

            if (this.baseSeries != null && this.currentSeries.concat(this.baseSeries).length > 0) {
                this.renderLineChart()
                this.renderSparkLines()
            } else if (this.missingBaseExperiment) {
                (new ErrorMessage('Base Experiment Not Found')).render(this.lineChartContainer)
            }
        } catch (e) {
            if (DEBUG) {
                console.error(e)
            }
        }
    }

    renderLineChart() {
        this.lineChartContainer.innerHTML = ''
        $(this.lineChartContainer, $ => {
            new CompareLineChart({
                series: this.currentSeries,
                baseSeries: this.baseSeries,
                width: this.width,
                currentPlotIdx: this.currentPlotIdx,
                basePlotIdx: this.basePlotIdx,
                chartType: this.preferenceData && this.preferenceData.chart_type ?
                    getChartType(this.preferenceData.chart_type) : 'linear',
                isDivergent: true
            }).render($)
        })
    }

    renderSparkLines() {
        this.sparkLinesContainer.innerHTML = ''
        $(this.sparkLinesContainer, $ => {
            new CompareSparkLines({
                series: this.currentSeries,
                baseSeries: this.baseSeries,
                currentPlotIdx: this.currentPlotIdx,
                basePlotIdx: this.basePlotIdx,
                width: this.width,
                isDivergent: true
            }).render($)
        })
    }

    async refresh() {
        try {
            await this.loader.load(true)
            if (this.currentSeries.concat(this.baseSeries).length > 0) {
                this.renderLineChart()
                this.renderSparkLines()
            }
        } catch (e) {
        }
    }

    onClick = () => {
        ROUTER.navigate(`/run/${this.currentUuid}/compare`)
    }
}
