import {Weya as $, WeyaElementFunction} from "../../../lib/weya/weya"
import {LogModel, Logs, LogUpdateType} from "../models/run";
import Filter from "../utils/ansi_to_html"
import {CustomButton} from "./buttons"

export class LogView {
    private logs: Logs
    private readonly filter: Filter
    private elem: HTMLDivElement
    private readonly logElems: Record<string, HTMLPreElement>
    private logElemLength: number

    private readonly loadPage:  (currentPage: number) => Promise<Logs>
    private readonly loadMoreButton: CustomButton
    private readonly loadAllButton: CustomButton
    private toLoadPage: number = -1

    constructor(logs: Logs, loadPage: (currentPage: number) => Promise<Logs>) {
        this.logs = logs
        this.logElems = {}
        this.logElemLength = 0
        this.filter = new Filter({})

        this.loadPage = loadPage

        this.loadMoreButton = new CustomButton({
            parent: this.constructor.name,
            text: "Load More",
            onButtonClick: this.onLoadMoreClick
        })
        this.loadAllButton = new CustomButton({
            parent: this.constructor.name,
            text: "Load All",
            onButtonClick: this.onLoadAllClick
        })
    }

    render($: WeyaElementFunction) {
        this.logElemLength = 0
        this.elem = $('div', '.std', $ => {
            this.loadAllButton
                .render($)
            this.loadMoreButton
                .render($)
        })
        this.renderLogs()
    }

    public invalidateLogs() {
        delete this.logs
        this.logs = new Logs(<LogModel>{pages: {}, page_length: 0})
    }

    public addLogs(logs: Logs) {
        this.logs.mergeLogs(logs)
        this.renderLogs()
    }

    private setLoading(isLoading: boolean) {
        this.loadMoreButton.disabled = isLoading
        this.loadAllButton.disabled = isLoading
    }

    private onLoadMoreClick = () => {
        this.setLoading(true)
        this.loadPage(this.toLoadPage).then((logs) => {
            this.logs.mergeLogs(logs)
            this.renderLogs()
            this.setLoading(false)
        })
    }

    private onLoadAllClick = () => {
        this.setLoading(true)
        this.loadPage(LogUpdateType.ALL).then((logs) => {
            this.logs.mergeLogs(logs)
            this.renderLogs()
            this.setLoading(false)
        })
    }

    private renderLogs() {
        while (this.logElemLength < this.logs.pageLength) {
            $(this.elem, $ => {
                this.logElems[this.logElemLength] = $('pre', '')
                this.logElemLength += 1
            })
        }

        let index: number
        for (index = this.logs.pageLength - 1; index >= 0; index--) {
            if (!this.logs.hasPage(index)) {
                this.toLoadPage = index
                this.loadMoreButton.hide(false)
                break
            }

            this.logElems[index].innerHTML = this.filter.toHtml(this.logs.getPage(index))
            this.logElems[index].classList.remove("hidden")
        }

        if (index == -1) { // all loaded
            this.loadMoreButton.hide(true)
            this.loadAllButton.hide(true)
        } else {
            this.loadAllButton.hide(false)
            this.loadMoreButton.hide(false)
        }

        for (index; index >= 0; index--) {
            this.logElems[index].classList.add("hidden")
        }
    }
}