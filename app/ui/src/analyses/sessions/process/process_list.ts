import {Weya as $, WeyaElementFunction} from "../../../../../lib/weya/weya"
import {PointValue, SeriesModel} from "../../../models/run"
import {ProcessModel} from "./types"
import {getExtent, getScale, getTimeScale, toDate, toPointValue} from "../../../components/charts/utils"
import d3 from "../../../d3"
import {DefaultLineGradient} from "../../../components/charts/chart_gradients"
import {formatFixed} from "../../../utils/value"
import {ROUTER} from "../../../app"
import {TimeSeriesFill, TimeSeriesPlot} from "../../../components/charts/timeseries/plot"
import {BadgeView} from "../../../components/badge"
import {SearchView} from "../../../components/search"

interface ProcessSparkLineOptions {
    width: number
    name: string
    color: string
    series: PointValue[]
    stepExtent: [Date, Date]
    barExtent: [number, number]
}

class ProcessSparkLine {
    width: number
    name: string
    color: string
    series: PointValue[]
    yScale: d3.ScaleLinear<number, number>
    xScale: d3.ScaleTime<number, number>
    barScale: d3.ScaleLinear<number, number>

    constructor(opt: ProcessSparkLineOptions) {
        this.width = opt.width
        this.series = opt.series
        this.name = opt.name
        this.color = opt.color

        this.yScale = getScale(getExtent([this.series], d => d.value, true), -25)
        this.xScale = getTimeScale(opt.stepExtent, this.width)

        this.barScale = getScale(opt.barExtent, this.width - 36)
    }

    render($) {
        const last = this.series[this.series.length - 1]

        $(`div.sparkline-list-item.list-group-item.d-inline-block`, $ => {
            $('div.sparkline-content', {style: {width: `${this.width}px`}}, $ => {
                $('svg.sparkline', {style: {width: `${this.width}px`}, height: 51}, $ => {
                    $('g', {transform: `translate(${0}, 30)`}, $ => {
                        new TimeSeriesFill({
                            series: this.series,
                            xScale: this.xScale,
                            yScale: this.yScale,
                            color: '#7f8c8d',
                            colorIdx: 9
                        }).render($)
                        new TimeSeriesPlot({
                            series: this.series,
                            xScale: this.xScale,
                            yScale: this.yScale,
                            color: '#7f8c8d'
                        }).render($)
                    })
                    $('line', '.wide-stroke', {
                        x1: `${this.width - this.barScale(last.smoothed)}`,
                        y1: "0",
                        x2: `${this.width}`,
                        y2: "0",
                        style: {stroke: this.color},
                        transform: `translate(${0}, 42)`
                    })
                    $('g', {transform: `translate(${2}, ${46})`}, $ => {
                        $('text', '.line-text', `${this.name}: `,)
                        $('text', '.line-text.line-end', `${formatFixed(last.smoothed, 3)}`, {
                            style: {fill: this.color},
                            transform: `translate(${this.width - 4}, ${0})`
                        })
                    })
                })
            })
        })
    }

}

export interface ProcessListItemOptions {
    item: ProcessModel
    stepExtent: [Date, Date]
    cpuBarExtent: [number, number]
    rssBarExtent: [number, number]
    width: number
    onClick: (elem: ProcessListItem) => void
}

class ProcessListItem {
    item: ProcessModel
    width: number
    elem: HTMLAnchorElement
    stepExtent: [Date, Date]
    cpuBarExtent: [number, number]
    rssBarExtent: [number, number]
    onClick: (evt: Event) => void

    constructor(opt: ProcessListItemOptions) {
        this.item = opt.item
        this.width = opt.width - 52
        this.stepExtent = opt.stepExtent
        this.cpuBarExtent = opt.cpuBarExtent
        this.rssBarExtent = opt.rssBarExtent
        this.onClick = (e: Event) => {
            e.preventDefault()
            opt.onClick(this)
        }
    }

    render($: WeyaElementFunction) {
        this.elem = $('a', '.list-item.list-group-item.list-group-item-action',
            {href: `/details/${this.item.process_id}`, on: {click: this.onClick}},
            $ => {
                $('div', '.process-item', $ => {
                    $('div', $ => {
                        $('span', this.item.name)
                        if (this.item.dead) {
                            new BadgeView({text: 'dead'}).render($)
                        }
                        $('div', '.pid', $ => {
                            $('span', '.title', 'PID: ')
                            $('span', `${this.item.pid}`)
                        })
                    })
                    new ProcessSparkLine({
                        width: this.width / 2,
                        series: this.item.cpu.series,
                        stepExtent: this.stepExtent,
                        barExtent: this.cpuBarExtent,
                        color: "#ffa600",
                        name: 'CPU'
                    }).render($)
                    new ProcessSparkLine({
                        width: this.width / 2,
                        series: this.item.rss.series,
                        stepExtent: this.stepExtent,
                        barExtent: this.rssBarExtent,
                        color: "#bc5090",
                        name: 'RAM'
                    }).render($)
                })
            })
    }
}

export interface ProcessListOptions {
    uuid: string
    items: ProcessModel[]
    width: number
}

export class ProcessList {
    uuid: string
    width: number
    stepExtent: [number, number]
    cpuBarExtent: [number, number]
    rssBarExtent: [number, number]
    searchQuery: string
    processListContainer: HTMLDivElement
    items: ProcessModel[]
    currentProcessList: ProcessModel[]

    constructor(opt: ProcessListOptions) {
        this.uuid = opt.uuid
        this.items = opt.items
        this.width = opt.width

        let rss: SeriesModel[] = []
        let cpu: SeriesModel[] = []
        for (let item of this.items) {
            item.cpu.series = toPointValue(item.cpu)
            cpu.push(item.cpu)
            item.rss.series = toPointValue(item.rss)
            rss.push(item.rss)
        }

        let series: SeriesModel[] = cpu.concat(rss)
        this.stepExtent = getExtent(series.map(s => s.series), d => d.step)

        this.cpuBarExtent = getExtent(cpu.map(s => s.series), d => d.value, true)
        this.rssBarExtent = getExtent(rss.map(s => s.series), d => d.value, true)

        this.searchQuery = ''
    }

    onclick(elem: ProcessListItem) {
        ROUTER.navigate(`session/${this.uuid}/process/${elem.item.process_id}`)
    }

    render($: WeyaElementFunction) {
        $('div', '.runs-list', $ => {
            new SearchView({onSearch: this.onSearch}).render($)
            $('svg', {style: {height: `${1}px`}}, $ => {
                new DefaultLineGradient().render($)
            })
            this.processListContainer = $('div', '.list.runs-list.list-group')
        })

        this.renderList()
    }

    onSearch = (query: string) => {
        this.searchQuery = query
        this.renderList()
    }

    processFilter = (process: ProcessModel, query: RegExp) => {
        let name = process.name.toLowerCase()
        let pid = process.pid.toString()

        return (name.search(query) !== -1 || pid.search(query) !== -1)
    }

    private renderList() {
        if (this.items.length > 0) {
            let re = new RegExp(this.searchQuery.toLowerCase(), 'g')
            this.currentProcessList = this.items.filter(process => this.processFilter(process, re))

            this.processListContainer.innerHTML = ''
            $(this.processListContainer, $ => {
                for (let i = 0; i < this.currentProcessList.length; i++) {
                    new ProcessListItem({
                        item: this.currentProcessList[i],
                        width: this.width,
                        stepExtent: [toDate(this.stepExtent[0]), toDate(this.stepExtent[1])],
                        cpuBarExtent: this.cpuBarExtent,
                        rssBarExtent: this.rssBarExtent,
                        onClick: this.onclick.bind(this)
                    }).render($)
                }
            })
        }
    }
}
