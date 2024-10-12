import {Weya as $, WeyaElementFunction} from '../../../../../lib/weya/weya'
import {Run} from "../../../models/run"
import CACHE, {RunCache} from "../../../cache/cache"
import {Card, CardOptions} from "../../types"
import {DataLoader} from "../../../components/loader"
import {Configs} from "./components"
import {ROUTER} from '../../../app'

export class RunConfigsCard extends Card {
    run: Run
    uuid: string
    width: number
    runCache: RunCache
    elem: HTMLDivElement
    configsContainer: HTMLDivElement
    private loader: DataLoader

    constructor(opt: CardOptions) {
        super(opt)

        this.uuid = opt.uuid
        this.width = opt.width
        this.runCache = CACHE.getRun(this.uuid)
        this.loader = new DataLoader(async (force) => {
            this.run = await this.runCache.get(force)
        })
    }

    cardName(): string {
        return 'Run Configs'
    }

    getLastUpdated(): number {
        return this.runCache.lastUpdated
    }

    async render($: WeyaElementFunction) {
        this.elem = $('div','.labml-card.labml-card-action', {on: {click: this.onClick}}, $ => {
            $('h3','.header', 'Configurations')
            this.loader.render($)
            this.configsContainer = $('div')
        })

        try {
            await this.loader.load()

            if (this.run.configs.length > 0) {
                this.renderConfigs()
            } else {
                this.elem.classList.add('hide')
            }
        } catch (e) {

        }
    }

    async refresh() {
        try {
            await this.loader.load(true)
            if (this.run.configs.length > 0) {
                this.renderConfigs()
                this.elem.classList.remove('hide')
            }
        } catch (e) {

        }
    }

    renderConfigs() {
        this.configsContainer.innerHTML = ''
        $(this.configsContainer, $ => {
            new Configs({configs: this.run.configs, width: this.width, isSummary: true}).render($)
        })
    }

    onClick = () => {
        ROUTER.navigate(`/run/${this.uuid}/configs`)
    }
}
