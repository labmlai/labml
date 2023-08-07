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
import {LineChart} from "../../../components/charts/lines/chart";

export class ComparisonCard extends Card {
    private readonly  currentUUID: string
    private baseUUID: string
    private width: number
    private baseAnalysisCache: AnalysisDataCache
    private baseSeries: SeriesModel[]
    private currentSeries: SeriesModel[]
    private currentAnalysisCache: AnalysisDataCache
    private currentPreferenceCache: AnalysisPreferenceCache
    private loader: DataLoader

    private lineChartContainer: HTMLDivElement
    private sparkLinesContainer: HTMLDivElement
    private elem: HTMLDivElement

    constructor(opt: CardOptions) {
        super(opt)

        this.currentUUID = opt.uuid
        this.width = opt.width
        this.currentAnalysisCache = comparisonCache.getAnalysis(this.currentUUID)
        this.currentPreferenceCache = comparisonCache.getPreferences(this.currentUUID)

        this.loader = new DataLoader(async (force: boolean) => {
            let preferenceData = <ComparisonPreferenceModel> await this.currentPreferenceCache.get(force)
            this.baseUUID = preferenceData.base_experiment

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
                width: this.width,
                plotIdx: [],
                chartType: 'linear',
                isDivergent: true
            }).render($)
            new LineChart({
                series: this.baseSeries,
                width: this.width,
                plotIdx: [],
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