import {Weya as $, WeyaElementFunction} from '../../../../../lib/weya/weya'
import {Run} from "../../../models/run"
import CACHE, {RunCache} from "../../../cache/cache"
import {Card, CardOptions} from "../../types"
import Filter from "../../../utils/ansi_to_html"
import {DataLoader} from "../../../components/loader"
import {ROUTER} from '../../../app';

export class StdErrorCard extends Card {
    run: Run
    uuid: string
    runCache: RunCache
    outputContainer: HTMLPreElement
    elem: HTMLDivElement
    filter: Filter
    private loader: DataLoader

    constructor(opt: CardOptions) {
        super(opt)

        this.uuid = opt.uuid
        this.runCache = CACHE.getRun(this.uuid)
        this.filter = new Filter({})
        this.loader = new DataLoader(async (force) => {
            this.run = await this.runCache.get(force)
        })
    }

    getLastTenLines(inputStr: string) {
        let split = inputStr.split("\n")

        let last10Lines
        if (split.length > 10) {
            last10Lines = split.slice(Math.max(split.length - 10, 1))
        } else {
            last10Lines = split
        }

        return last10Lines.join("\n")
    }

    getLastUpdated(): number {
        return this.runCache.lastUpdated
    }

    async render($: WeyaElementFunction) {
        this.elem = $('div', '.labml-card.labml-card-action', {on: {click: this.onClick}}, $ => {
            $('h3', '.header', 'Standard Error')
            this.loader.render($)
            $('div', '.terminal-card.no-scroll', $ => {
                this.outputContainer = $('pre', '')
            })
        })

        try {
            await this.loader.load()

            if (this.run.stderr) {
                this.renderOutput()
            } else {
                this.elem.classList.add('hide')
            }
        } catch (e) {

        }
    }

    renderOutput() {
        this.outputContainer.innerHTML = ''
        $(this.outputContainer, $ => {
            let output = $('div', '')
            output.innerHTML = this.filter.toHtml(this.getLastTenLines(this.run.stderr))
        })
    }

    async refresh() {
        try {
            await this.loader.load(true)
            if (this.run.stderr) {
                this.renderOutput()
                this.elem.classList.remove('hide')
            }
        } catch (e) {

        }
    }

    onClick = () => {
        ROUTER.navigate(`/run/${this.uuid}/stderr`)
    }
}
