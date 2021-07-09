import {SeriesModel} from "../../../models/run"
import {AnalysisDataCache} from "../../../cache/cache"
import {Weya as $, WeyaElementFunction} from "../../../../../lib/weya/weya"
import {DataLoader} from "../../../components/loader"
import {Card, CardOptions} from "../../types"
import gpuCache from "./cache"
import {getSeriesData} from "./utils"
import {Labels} from "../../../components/charts/labels"
import {TimeSeriesChart} from "../../../components/charts/timeseries/chart"
import {ROUTER} from "../../../app"

export class GPUMemoryCard extends Card {
    uuid: string
    width: number
    series: SeriesModel[]
    analysisCache: AnalysisDataCache
    lineChartContainer: HTMLDivElement
    elem: HTMLDivElement
    plotIdx: number[] = []
    private loader: DataLoader
    private labelsContainer: HTMLDivElement

    constructor(opt: CardOptions) {
        super(opt)

        this.uuid = opt.uuid
        this.width = opt.width
        this.analysisCache = gpuCache.getAnalysis(this.uuid)
        this.loader = new DataLoader(async (force) => {
            this.series = getSeriesData((await this.analysisCache.get(force)).series, 'memory')
        })
    }

    getLastUpdated(): number {
        return this.analysisCache.lastUpdated
    }

    async render($: WeyaElementFunction) {
        this.elem = $('div', '.labml-card.labml-card-action', {on: {click: this.onClick}}, $ => {
            $('h3', '.header', 'GPU - Memory')
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
        let res: number[] = []
        for (let i = 0; i < this.series.length; i++) {
            res.push(i)
        }
        this.plotIdx = res

        this.lineChartContainer.innerHTML = ''
        $(this.lineChartContainer, $ => {
            new TimeSeriesChart({
                series: this.series,
                width: this.width,
                plotIdx: this.plotIdx,
                chartHeightFraction: 4,
                isDivergent: true
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
        ROUTER.navigate(`/session/${this.uuid}/gpu_memory`)
    }
}
