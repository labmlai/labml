import {Weya as $, WeyaElementFunction} from '../../../../../lib/weya/weya'
import {Card, CardOptions} from "../../types"
import Filter from "../../../utils/ansi_to_html"
import {DataLoader} from "../../../components/loader"
import {ROUTER} from '../../../app'
import stdOutCache from "./cache";
import {Logs} from "../../../models/run"

export class StdOutCard extends Card {
    uuid: string
    outputContainer: HTMLPreElement
    elem: HTMLDivElement
    filter: Filter
    private loader: DataLoader
    private stdOut?: Logs

    constructor(opt: CardOptions) {
        super(opt)

        this.uuid = opt.uuid
        this.filter = new Filter({})
        this.loader = new DataLoader(async (force) => {
            this.stdOut = await stdOutCache.getLogCache(this.uuid).getLast(force)
        })
    }

    getLastTenLines(inputStr: string) {
        let split = inputStr.split("\n")

        let last10Lines: string[]
        if (split.length > 10) {
            last10Lines = split.slice(Math.max(split.length - 10, 1))
        } else {
            last10Lines = split
        }

        return last10Lines.join("\n")
    }

    getLastUpdated(): number {
        return stdOutCache.getLogCache(this.uuid).lastUpdated
    }

    async render($: WeyaElementFunction) {
        this.elem = $('div', '.labml-card.labml-card-action', {on: {click: this.onClick}}, $ => {
            $('h3', '.header', 'Standard Output')
            this.loader.render($)
            $('div', '.terminal-card.no-scroll', $ => {
                this.outputContainer = $('pre', '')
            })
        })

        try {
            await this.loader.load()

            if (this.stdOut?.hasPage(this.stdOut?.pageLength - 1)) {
                this.renderOutput()
                this.elem.classList.remove('hide')
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
            output.innerHTML = this.filter.toHtml(this.getLastTenLines(this.stdOut.getPage(this.stdOut.pageLength - 1)))
        })
    }

    async refresh() {
        try {
            await this.loader.load(true)
            if (this.stdOut?.hasPage(this.stdOut?.pageLength - 1)) {
                this.renderOutput()
                this.elem.classList.remove('hide')
            } else {
                this.elem.classList.add('hide')
            }
        } catch (e) {

        }
    }

    onClick = () => {
        ROUTER.navigate(`/run/${this.uuid}/stdout`)
    }
}
