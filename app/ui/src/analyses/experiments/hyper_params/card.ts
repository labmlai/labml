import {Weya, WeyaElement, WeyaElementFunction,} from '../../../../../lib/weya/weya'
import {SeriesModel} from "../../../models/run"
import {Card, CardOptions} from "../../types"
import {AnalysisDataCache, AnalysisPreferenceCache,} from "../../../cache/cache"
import hyperParamsCache from "./cache"
import {DataLoader} from "../../../components/loader"
import {ROUTER} from '../../../app'
import {toPointValues} from "../../../components/charts/utils"
import {SparkLines} from "../../../components/charts/spark_lines/chart"
import {AnalysisPreferenceModel} from "../../../models/preferences"


export class HyperParamsCard extends Card {
    uuid: string
    width: number
    series: SeriesModel[]
    preferenceData: AnalysisPreferenceModel
    analysisCache: AnalysisDataCache
    preferenceCache: AnalysisPreferenceCache
    elem: WeyaElement
    sparkLinesContainer: WeyaElement
    plotIdx: number[] = []
    private loader: DataLoader

    constructor(opt: CardOptions) {
        super(opt)

        this.uuid = opt.uuid
        this.width = opt.width
        this.analysisCache = hyperParamsCache.getAnalysis(this.uuid)
        this.preferenceCache = hyperParamsCache.getPreferences(this.uuid)

        this.loader = new DataLoader(async (force) => {
            this.series = toPointValues((await this.analysisCache.get(force)).series)
            this.preferenceData = await this.preferenceCache.get(force)

            let res: number[] = []
            for (let i = 0; i < this.series.length; i++) {
                res.push(i)
            }
            this.plotIdx = res
        })
    }

    getLastUpdated(): number {
        return this.analysisCache.lastUpdated
    }

    async render($: WeyaElementFunction) {
        this.elem = $('div', '.labml-card.labml-card-action', {on: {click: this.onClick}}, $ => {
            $('h3','.header', 'Hyper-Parameters')
            this.loader.render($)
            this.sparkLinesContainer = $('div', '')
        })

        try {
            await this.loader.load()

            let analysisPreferences = this.preferenceData.series_preferences
            if (analysisPreferences.length > 0) {
                this.plotIdx = [...analysisPreferences]
            }

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
        Weya(this.sparkLinesContainer, $ => {
            new SparkLines({
                series: this.series,
                plotIdx: this.plotIdx,
                width: this.width,
                isDivergent: true
            }).render($)
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
        ROUTER.navigate(`/run/${this.uuid}/hyper_params`)
    }
}

