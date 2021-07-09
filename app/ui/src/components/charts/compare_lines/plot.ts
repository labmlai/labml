import d3 from "../../../d3"
import {WeyaElement, WeyaElementFunction} from '../../../../../lib/weya/weya'
import {FillOptions, PlotOptions} from '../types'
import {PointValue} from "../../../models/run"
import {getSelectedIdx} from "../utils"

export interface CompareLinePlotOptions extends PlotOptions {
    xScale: d3.ScaleLinear<number, number>
    series: PointValue[]
    isDotted?: boolean
}

export class CompareLinePlot {
    series: PointValue[]
    xScale: d3.ScaleLinear<number, number>
    yScale: d3.ScaleLinear<number, number>
    color: string
    circleElem: WeyaElement
    smoothedLine: d3.Line<PointValue>
    unsmoothedLine: d3.Line<PointValue>
    bisect: d3.Bisector<number, number>
    isDotted: boolean

    constructor(opt: CompareLinePlotOptions) {
        this.series = opt.series
        this.xScale = opt.xScale
        this.yScale = opt.yScale
        this.color = opt.color
        this.isDotted = opt.isDotted ?? false

        this.bisect = d3.bisector(function (d: PointValue) {
            return d.step
        }).left

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
            $('path.smoothed-line.dropshadow',
                {
                    fill: 'none',
                    stroke: this.color,
                    d: this.smoothedLine(this.series) as string,
                    "stroke-dasharray": this.isDotted ? 5 : 0
                })
            if (!this.isDotted) {
                $('path.unsmoothed-line',
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
        })
    }

    renderCursorCircle(cursorStep: number | null) {
        if (cursorStep != null) {
            let idx = getSelectedIdx(this.series, this.bisect, cursorStep)

            this.circleElem.setAttribute("cx", `${this.xScale(this.series[idx].step)}`)
            this.circleElem.setAttribute("cy", `${this.yScale(this.series[idx].smoothed)}`)
            this.circleElem.setAttribute("r", `5`)
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
