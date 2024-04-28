import {Weya as $, WeyaElementFunction} from "../../../lib/weya/weya"
import {Logs} from "../models/run";
import Filter from "../utils/ansi_to_html"

export class LogView {
    private logs: Logs
    private readonly filter: Filter
    private elem: HTMLDivElement
    private logElems: Record<string, HTMLPreElement>
    private logElemLength: number

    private readonly loadPage:  (currentPage: number) => Promise<Logs>

    constructor(logs: Logs, loadPage: (currentPage: number) => Promise<Logs>) {
        this.logs = logs
        this.logElems = {}
        this.logElemLength = 0
        this.filter = new Filter({})

        this.loadBelow.bind(this)
        this.loadAbove.bind(this)

        this.loadPage = loadPage
    }

    render($: WeyaElementFunction) {
        this.elem = $('div')
        this.renderLogs()
    }

    public addLogs(logs: Logs) {
        this.logs.mergeLogs(logs)
        this.renderLogs()
    }

    private loadAbove(currentPage: number) {
        this.loadPage(currentPage).then((logs) => {
            this.logs.mergeLogs(logs)
            this.renderLogs()
        })
    }

    private loadBelow(currentPage: number) {
        this.loadPage(currentPage).then((logs) => {
            this.logs.mergeLogs(logs)
            this.renderLogs()
        })
    }

    private renderLogs() {
        while (this.logElemLength < this.logs.pageLength) {
            $(this.elem, $ => {
                this.logElems[this.logElemLength] = $('pre', '')
                this.logElemLength += 1
            })
        }

        let lastPage = -2
        for (let i = 0; i < this.logs.pageLength; i++) {
            let content = this.logs.getPage(i)
            if (content != null) {
                if (lastPage != i-1 && i != 0) {
                    this.logElems[i-1].innerHTML = 'Load more above...'
                    this.logElems[i-1].removeEventListener('click', () => this.loadBelow(i-1))
                    this.logElems[i-1].addEventListener('click', () => this.loadAbove(i-1))
                }

                // if (this.logElems[i].innerHTML != "") { // already rendered
                //     continue
                // }

                this.logElems[i].innerHTML = this.filter.toHtml(this.logs.getPage(i))

                // remove listeners
                this.logElems[i].removeEventListener('click', () => this.loadBelow(i))
                this.logElems[i].removeEventListener('click', () => this.loadAbove(i))

                lastPage = i
            } else if (lastPage == i-1) {
                this.logElems[i].innerHTML = 'Load more below...'
                this.logElems[i].removeEventListener('click', () => this.loadAbove(i))
                this.logElems[i].addEventListener('click', () => this.loadBelow(i))
            } else {
                this.logElems[i].innerHTML = ''
                this.logElems[i].removeEventListener('click', () => this.loadBelow(i))
                this.logElems[i].removeEventListener('click', () => this.loadAbove(i))
            }
        }
    }
}