import {Weya as $, WeyaElementFunction,} from '../../../../../lib/weya/weya'
import {Card, CardOptions} from "../../types"
import {ProcessDataCache} from "../../../cache/cache"
import {DataLoader} from "../../../components/loader"
import {processCache} from './cache'
import {ROUTER} from '../../../app'
import {SparkTimeLines} from "../../../components/charts/spark_time_lines/chart"
import {Indicator} from "../../../models/run";

export class ProcessCard extends Card {
    uuid: string
    width: number
    series: Indicator[]
    analysisCache: ProcessDataCache
    sparkLinesContainer: HTMLDivElement
    sparkTimeLines: SparkTimeLines
    elem: HTMLDivElement
    private loader: DataLoader

    constructor(opt: CardOptions) {
        super(opt)

        this.uuid = opt.uuid
        this.width = opt.width
        this.analysisCache = processCache.getAnalysis(this.uuid)
        this.loader = new DataLoader(async (force) => {
            this.series = (await this.analysisCache.get(force)).summary
        })
    }

    cardName(): string {
        return 'Process'
    }

    getLastUpdated(): number {
        return this.analysisCache.lastUpdated
    }

    async render($: WeyaElementFunction) {
        this.elem = $('div', '.labml-card.labml-card-action', {on: {click: this.onClick}}, $ => {
            $('h3', '.header', 'Processes')
            this.loader.render($)
            this.sparkLinesContainer = $('div', '')
        })

        try {
            await this.loader.load()

            if (this.series.length > 0) {
                this.renderSparkLines()
            } else {
                this.elem.classList.add('hide')
            }
        } catch (e) {

        }
    }

    renderSparkLines() {
        this.sparkLinesContainer.innerHTML = ''
        $(this.sparkLinesContainer, $ => {
            this.sparkTimeLines = new SparkTimeLines({
                series: this.series,
                plotIdx: [1, 2, 3, 4, 5],
                width: this.width,
                isColorless: true
            })
            this.sparkTimeLines.render($)
        })
    }

    async refresh() {
        try {
            await this.loader.load(true)
            if (this.series.length > 0) {
                this.renderSparkLines()
                this.elem.classList.remove('hide')
            }
        } catch (e) {

        }
    }

    onClick = () => {
        ROUTER.navigate(`/session/${this.uuid}/process`)
    }
}
