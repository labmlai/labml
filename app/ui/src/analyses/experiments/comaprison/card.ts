import {Card, CardOptions} from "../../types"
import {Weya as $, WeyaElementFunction} from "../../../../../lib/weya/weya"
import {AnalysisDataCache, AnalysisPreferenceCache} from "../../../cache/cache"
import comparisonCache from "./cache"
import {DataLoader} from "../../../components/loader"
import {ComparisonPreferenceModel} from "../../../models/preferences"
import {DEBUG} from "../../../env"
import {clearChildElements} from "../../../utils/document"
import {toPointValues} from "../../../components/charts/utils"
import {SeriesModel} from "../../../models/run"
import {LineChart} from "../../../components/charts/compare_lines/chart_new"

export class ComparisonCard extends Card {
    private readonly  currentUUID: string
    private baseUUID: string
    private width: number
    private baseAnalysisCache: AnalysisDataCache
    private baseSeries: SeriesModel[]
    private currentSeries: SeriesModel[]
    private currentAnalysisCache: AnalysisDataCache
    private preferenceCache: AnalysisPreferenceCache
    private preferenceData: ComparisonPreferenceModel
    private loader: DataLoader

    private lineChartContainer: HTMLDivElement
    private sparkLinesContainer: HTMLDivElement
    private elem: HTMLDivElement

    constructor(opt: CardOptions) {
        super(opt)

        this.currentUUID = opt.uuid
        this.width = opt.width
        this.currentAnalysisCache = comparisonCache.getAnalysis(this.currentUUID)
        this.preferenceCache = comparisonCache.getPreferences(this.currentUUID)

        this.loader = new DataLoader(async (force: boolean) => {
            this.preferenceData = <ComparisonPreferenceModel> await this.preferenceCache.get(force)
            this.baseUUID = this.preferenceData.base_experiment

            let currentAnalysisData = await this.currentAnalysisCache.get(force)
            this.currentSeries = toPointValues(currentAnalysisData.series)
            if (!!this.baseUUID) {
                this.baseAnalysisCache = comparisonCache.getAnalysis(this.baseUUID)
                try {
                    let baseAnalysisData = await this.baseAnalysisCache.get(force)
                    this.baseSeries = toPointValues(baseAnalysisData.series)
                } catch (e) {
                    // TODO handle series error
                }
            }
        })

    }

    getLastUpdated(): number {
        return this.currentAnalysisCache.lastUpdated;
    }

    refresh() {
        throw new Error("Not implemented")
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

            this.renderLineChart()
            this.renderSparkLines()
        } catch (e) {
            if (DEBUG) {
                console.error(e)
            }
        }
    }

    private renderLineChart() {
        clearChildElements(this.lineChartContainer)
        $(this.lineChartContainer, $ => {
            new LineChart({
                series: this.currentSeries,
                baseSeries: this.baseSeries,
                currentPlotIndex: [...(this.preferenceData.series_preferences ?? [])],
                basePlotIdx: [...(this.preferenceData.base_series_preferences ?? [])],
                width: this.width,
                chartType: 'linear',
                isDivergent: true
            }).render($)
        })
    }

    private renderSparkLines() {
        clearChildElements(this.sparkLinesContainer)
    }

    onClick = () => {
       throw new Error("Not implemented")
    }
}