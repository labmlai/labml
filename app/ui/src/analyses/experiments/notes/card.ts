import {Weya as $, WeyaElementFunction} from '../../../../../lib/weya/weya'
import CACHE, {RunCache} from "../../../cache/cache"
import {Card, CardOptions} from "../../types"
import {DataLoader} from "../../../components/loader"
import {ROUTER} from '../../../app'
import {marked} from 'marked'

export class NotesCard extends Card {
    runCache: RunCache
    outputContainer: HTMLElement
    elem: HTMLDivElement
    private loader: DataLoader
    notes: string
    uuid: string

    constructor(opt: CardOptions) {
        super(opt)

        this.uuid = opt.uuid

        this.runCache = CACHE.getRun(opt.uuid)
        this.loader = new DataLoader(async (force) => {
            this.notes = (await this.runCache.get(force)).note
        })
    }

    cardName(): string {
        return 'Logger'
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
        return this.runCache.lastUpdated
    }

    async render($: WeyaElementFunction) {
        this.elem = $('div', '.labml-card.labml-card-action', {on: {click: this.onClick}}, $ => {
            $('h3', '.header', 'Notes')
            this.loader.render($)
            this.outputContainer = $('div', '.md-notes', $ => {
            })
        })

        try {
            await this.loader.load()

            if (this.notes != "") {
                this.renderNotes()
                this.elem.classList.remove('hide')
            } else {
                this.elem.classList.add('hide')
            }
        } catch (e) {

        }
    }

    renderNotes() {
        this.outputContainer.innerHTML = ''
        $(this.outputContainer, async $ => {
            let output = $('div', '')
            output.innerHTML = await marked(this.notes)
        })
    }

    async refresh() {
        try {
            await this.loader.load(true)
            if (this.notes != "") {
                this.renderNotes()
                this.elem.classList.remove('hide')
            } else {
                this.elem.classList.add('hide')
            }
        } catch (e) {

        }
    }

    onClick = () => {
        ROUTER.navigate(`/run/${this.uuid}/header`)
    }
}
