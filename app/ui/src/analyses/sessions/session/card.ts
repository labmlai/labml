import {Weya as $, WeyaElementFunction,} from '../../../../../lib/weya/weya'
import {SeriesModel} from "../../../models/run"
import {Card, CardOptions} from "../../types"
import {AnalysisDataCache, AnalysisPreferenceCache} from "../../../cache/cache"
import {toPointValues} from "../../../components/charts/utils"
import {DataLoader} from "../../../components/loader"
import {TimeSeriesChart} from "../../../components/charts/timeseries/chart"
import {Labels} from "../../../components/charts/labels"
import {ROUTER} from '../../../app'
import {AnalysisCache} from "../../helpers"
import {getSeriesData} from "../gpu/utils"
import {AnalysisPreferenceModel} from "../../../models/preferences"

export class SessionCard extends Card {
    uuid: string
    width: number
    series: SeriesModel[]
    analysisCache: AnalysisDataCache
    preferenceCache: AnalysisPreferenceCache
    preferenceData: AnalysisPreferenceModel
    lineChartContainer: HTMLDivElement
    elem: HTMLDivElement
    private loader: DataLoader
    private labelsContainer: HTMLDivElement
    private readonly title: string
    private readonly url: string
    private readonly yExtend?: [number, number]
    private plotIndex: number[]
    private readonly subSeries?: string

    constructor(opt: CardOptions,
                title: string,
                url: string,
                cache: AnalysisCache<any, any>,
                plotIndex: number[],
                isSummary: boolean,
                yExtend?: [number, number], subSeries?: string) {
        super(opt)

        this.uuid = opt.uuid
        this.width = opt.width
        this.title = title
        this.url = url
        this.yExtend = yExtend
        this.subSeries = subSeries
        this.analysisCache = cache.getAnalysis(this.uuid)
        this.preferenceCache = cache.getPreferences(this.uuid)
        this.plotIndex = plotIndex
        this.loader = new DataLoader(async (force) => {
            if (isSummary) {
                let data: any[] = (await this.analysisCache.get(force)).summary
                this.series = toPointValues(data)
            } else {
                if (this.subSeries) {
                    this.series = getSeriesData((await this.analysisCache.get(force)).series, this.subSeries)
                } else {
                    let data: any[] = (await this.analysisCache.get(force)).series
                    this.series = toPointValues(data)
                }
            }
            this.preferenceData = await this.preferenceCache.get(force)
            if (this.subSeries) { // show all plots of sub series
                this.plotIndex = []
                let i = 0
                for (let _ of this.series) {
                    this.plotIndex.push(i++)
                }
            }
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
                stepRange: this.preferenceData.step_range,
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
