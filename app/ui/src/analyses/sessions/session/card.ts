import {Weya as $, WeyaElementFunction,} from '../../../../../lib/weya/weya'
import {SeriesModel} from "../../../models/run"
import {Card, CardOptions} from "../../types"
import {AnalysisDataCache} from "../../../cache/cache"
import {toPointValues} from "../../../components/charts/utils"
import {DataLoader} from "../../../components/loader"
import {TimeSeriesChart} from "../../../components/charts/timeseries/chart"
import {Labels} from "../../../components/charts/labels"
import {ROUTER} from '../../../app'
import {AnalysisCache} from "../../helpers"

export class SessionCard extends Card {
    uuid: string
    width: number
    series: SeriesModel[]
    analysisCache: AnalysisDataCache
    lineChartContainer: HTMLDivElement
    elem: HTMLDivElement
    private loader: DataLoader
    private labelsContainer: HTMLDivElement
    private readonly title: string
    private readonly url: string
    private readonly yExtend?: [number, number]
    private readonly plotIndex: number[]

    constructor(opt: CardOptions,
                title: string,
                url: string,
                cache: AnalysisCache<any, any>,
                plotIndex: number[],
                isSummary: boolean,
                yExtend?: [number, number],) {
        super(opt)

        this.uuid = opt.uuid
        this.width = opt.width
        this.title = title
        this.url = url
        this.yExtend = yExtend
        this.analysisCache = cache.getAnalysis(this.uuid)
        this.plotIndex = plotIndex
        this.loader = new DataLoader(async (force) => {
            let data: any[]
                if (isSummary) {
                    data = (await this.analysisCache.get(force)).summary
                } else {
                    data = (await this.analysisCache.get(force)).series
                }
            this.series = toPointValues(data)
        })
    }

    getLastUpdated(): number {
        return this.analysisCache.lastUpdated
    }

    async render($: WeyaElementFunction) {
        this.elem = $('div', '.labml-card.labml-card-action', {on: {click: this.onClick}}, $ => {
            $('h3', '.header', this.title)
            this.loader.render($)
            this.lineChartContainer = $('div', '')
            this.labelsContainer = $('div', '')
        })

        try {
            await this.loader.load()

            if (this.series.length > 0) {
                this.renderLineChart()
            } else {
                this.elem.classList.add('hide')
            }
        } catch (e) {

        }
    }

    renderLineChart() {
        this.lineChartContainer.innerHTML = ''
        $(this.lineChartContainer, $ => {
            new TimeSeriesChart({
                series: this.series,
                width: this.width,
                plotIdx: this.plotIndex,
                yExtend: this.yExtend,
                chartHeightFraction: 4,
                isDivergent: true,
            }).render($)
        })

        this.labelsContainer.innerHTML = ''
        $(this.labelsContainer, $ => {
            new Labels({labels: Array.from(this.series, x => x['name']), isDivergent: true}).render($)
        })
    }

    async refresh() {
        try {
            await this.loader.load(true)
            if (this.series.length > 0) {
                this.renderLineChart()
                this.elem.classList.remove('hide')
            }
        } catch (e) {

        }
    }

    onClick = () => {
        ROUTER.navigate(`/session/${this.uuid}/${this.url}`)
    }
}
