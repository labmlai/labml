import {Weya as $, WeyaElementFunction,} from '../../../../../lib/weya/weya'
import {AnalysisDataModel} from "../../../models/run"
import {Card, CardOptions} from "../../types"
import {AnalysisDataCache} from "../../../cache/cache"
import {SimpleLinesChart} from "../../../components/charts/simple_lines/chart"
import parametersCache from "./cache"
import {DataLoader} from "../../../components/loader"
import {ROUTER} from '../../../app'

export class ParametersCard extends Card {
    uuid: string
    width: number
    analysisData: AnalysisDataModel
    analysisCache: AnalysisDataCache
    elem: HTMLDivElement
    lineChartContainer: HTMLDivElement
    private loader: DataLoader

    constructor(opt: CardOptions) {
        super(opt)

        this.uuid = opt.uuid
        this.width = opt.width
        this.analysisCache = parametersCache.getAnalysis(this.uuid)
        this.loader = new DataLoader(async (force) => {
            this.analysisData = await this.analysisCache.get(force)
        })
    }

    getLastUpdated(): number {
        return this.analysisCache.lastUpdated
    }

    async render($: WeyaElementFunction) {
        this.elem = $('div', '.labml-card.labml-card-action', {on: {click: this.onClick}}, $ => {
            $('h3', '.header', 'Parameters')
            this.loader.render($)
            this.lineChartContainer = $('div', '')
        })

        try {
            await this.loader.load()

            if (this.analysisData.summary.length > 0) {
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
            new SimpleLinesChart({series: this.analysisData.summary, width: this.width}).render($)
        })
    }

    async refresh() {
        try {
            await this.loader.load(true)

            if (this.analysisData.summary.length > 0) {
                this.renderLineChart()
                this.elem.classList.remove('hide')
            }
        } catch (e) {

        }
    }

    onClick = () => {
        ROUTER.navigate(`/run/${this.uuid}/params`)
    }
}
