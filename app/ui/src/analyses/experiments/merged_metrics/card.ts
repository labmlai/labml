import {WeyaElement, WeyaElementFunction,} from '../../../../../lib/weya/weya'
import {InsightModel, SeriesModel} from "../../../models/run"
import {AnalysisPreferenceModel} from "../../../models/preferences"
import {Card, CardOptions} from "../../types"
import {toPointValues} from "../../../components/charts/utils"
import {ROUTER} from '../../../app'
import {DataLoader} from '../../../components/loader'
import {CardWrapper} from "../chart_wrapper/card"
import metricsCache from "./cache"


export class DistributedMetricsCard extends Card {
    private readonly uuid: string
    private readonly width: number
    private series: SeriesModel[]
    private insights: InsightModel[]
    private preferenceData: AnalysisPreferenceModel
    private elem: HTMLDivElement
    private lineChartContainer: WeyaElement
    private insightsContainer: WeyaElement
    private loader: DataLoader
    private chartWrapper: CardWrapper
    private sparkLineContainer: WeyaElement

    constructor(opt: CardOptions) {
        super(opt)

        this.uuid = opt.uuid
        this.width = opt.width
        this.loader = new DataLoader(async (force) => {
            let analysisData = await  metricsCache.getAnalysis(this.uuid).get(force)
            this.series = toPointValues(analysisData.series)
            this.insights = analysisData.insights
            this.preferenceData = await metricsCache.getPreferences(this.uuid).get(force)
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

            this.chartWrapper = new CardWrapper({
                elem: this.elem,
                preferenceData: this.preferenceData,
                insights: this.insights,
                series: this.series,
                insightsContainer: this.insightsContainer,
                lineChartContainer: this.lineChartContainer,
                sparkLinesContainer: this.sparkLineContainer,
                width: this.width,
                isDistributed: false
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
        ROUTER.navigate(`/run/${this.uuid}/merged_distributed`)
    }
}