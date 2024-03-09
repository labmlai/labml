import d3 from "../../../d3"
import {WeyaElement, WeyaElementFunction} from '../../../../../lib/weya/weya'
import {FillOptions, PlotOptions} from '../types'
import {PointValue} from "../../../models/run"
import {getSelectedIdx, mapRange, smoothSeries} from "../utils"

export interface LinePlotOptions extends PlotOptions {
    xScale: d3.ScaleLinear<number, number>
    series: PointValue[]
    isBase?: boolean
    renderHorizontalLine?: boolean
    smoothFocused?: boolean
    smoothedSeries: PointValue[]
}

export class LinePlot {
    series: PointValue[]
    xScale: d3.ScaleLinear<number, number>
    yScale: d3.ScaleLinear<number, number>
    color: string
    circleElem: WeyaElement
    lineElem: WeyaElement
    smoothedLine: d3.Line<PointValue>
    unsmoothedLine: d3.Line<PointValue>
    bisect: d3.Bisector<number, number>
    isBase: boolean
    renderHorizontalLine: boolean
    smoothFocused: boolean

    private readonly smoothedSeries: PointValue[]

    constructor(opt: LinePlotOptions) {
        this.series = opt.series
        this.xScale = opt.xScale
        this.yScale = opt.yScale
        this.color = opt.color
        this.isBase = opt.isBase ?? false
        this.renderHorizontalLine = opt.renderHorizontalLine ?? false
        this.smoothFocused = opt.smoothFocused ?? false

        this.bisect = d3.bisector(function (d: PointValue) {
            return d.step
        }).left

        this.smoothedSeries = opt.smoothedSeries

        this.smoothedLine = d3.line()
            .curve(d3.curveMonotoneX)
            .x((d, i) => {
                return this.xScale(d.step)
            })
            .y((d) => {
                return this.yScale(d.smoothed)
            })

        this.unsmoothedLine = d3.line()
            .curve(d3.curveMonotoneX)
            .x((d, i) => {
                return this.xScale(d.step)
            })
            .y((d) => {
                return this.yScale(d.value)
            })
    }

    render($: WeyaElementFunction) {
        $('g', $ => {
            $(`path.smoothed-line.dropshadow${this.isBase ? '.base': '.current'}`,
                {
                    fill: 'none',
                    stroke: this.color,
                    d: this.smoothedLine(this.smoothedSeries) as string,
                    "stroke-dasharray": this.isBase ? "3 1": ""
                })
            if (!this.isBase) {
                $('path.unsmoothed-line'+(this.smoothFocused ? '.smooth-focused': ''),
                    {
                        fill: 'none',
                        stroke: this.color,
                        d: this.unsmoothedLine(this.series) as string
                    })
            }
            $('g', $ => {
                this.circleElem = $('circle',
                    {
                        fill: this.color
                    })
            })
            $('g', $ => {
                this.lineElem = $('line',
                    {
                        stroke: this.color
                    })
            })
        })
    }

    renderIndicators(cursorStep: number | null) {
        this.renderCircle(cursorStep)
        if (this.renderHorizontalLine) {
            this.renderLine(cursorStep)
        }
    }

    private renderCircle(cursorStep: number | null) {
        if (cursorStep != null) {
            let idx = getSelectedIdx(this.smoothedSeries, this.bisect, cursorStep)

            if (idx == -1)
                return

            this.circleElem.setAttribute("cx", `${this.xScale(this.smoothedSeries[idx].step)}`)
            this.circleElem.setAttribute("cy", `${this.yScale(this.smoothedSeries[idx].smoothed)}`)
            this.circleElem.setAttribute("r", `5`)
        }
    }

    private renderLine(cursorStep: number | null) {
        if (cursorStep != null) {
            let idx = getSelectedIdx(this.smoothedSeries, this.bisect, cursorStep)

            if (idx == -1) {
                return
            }

            this.lineElem.setAttribute("x1", `${this.xScale(this.xScale.domain()[0])}`)
            this.lineElem.setAttribute("x2", `${this.xScale(this.xScale.domain()[1])}`)
            this.lineElem.setAttribute("y1", `${this.yScale(this.smoothedSeries[idx].smoothed).toFixed(2)}`)
            this.lineElem.setAttribute("y2", `${this.yScale(this.smoothedSeries[idx].smoothed).toFixed(2)}`)
            this.lineElem.setAttribute("stroke-width", `1`)
            this.lineElem.setAttribute("opacity", `0.5`)
        }
    }
}

interface LineFillOptions extends FillOptions {
    xScale: d3.ScaleLinear<number, number>
    series: PointValue[]
    chartId?: string
}

export class LineFill {
    series: PointValue[]
    chartId: string
    xScale: d3.ScaleLinear<number, number>
    yScale: d3.ScaleLinear<number, number>
    color: string
    colorIdx: number
    smoothedLine
    fill: string
    dFill: string

    constructor(opt: LineFillOptions) {
        this.series = opt.series
        this.xScale = opt.xScale
        this.yScale = opt.yScale
        this.color = opt.color
        this.colorIdx = opt.colorIdx
        this.chartId = opt.chartId

        this.smoothedLine = d3.line()
            .curve(d3.curveMonotoneX)
            .x((d, i) => {
                return this.xScale(d.step)
            })
            .y((d) => {
                return this.yScale(d.smoothed)
            })

        let d = this.smoothedLine(this.series) as string
        this.dFill = `M${this.xScale(this.series[0].step)},0L` + d.substr(1) +
            `L${this.xScale(this.series[this.series.length - 1].step)},0`

        this.fill = this.chartId ? `url(#gradient-${this.colorIdx}-${this.chartId}` : `url(#gradient-grey)`
    }

    render($: WeyaElementFunction) {
        $('g', $ => {
            $('path.line-fill',
                {
                    fill: this.color,
                    stroke: 'none',
                    style: {fill: this.fill},
                    d: this.dFill
                })
        })
    }
}
