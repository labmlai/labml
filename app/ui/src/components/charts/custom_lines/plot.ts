import d3 from "../../../d3"
import {WeyaElement, WeyaElementFunction} from '../../../../../lib/weya/weya'
import {FillOptions, PlotOptions} from '../types'
import {PointValue} from "../../../models/run"
import {getSelectedIdx} from "../utils"


export interface LinePlotOptions extends PlotOptions {
    xScale: d3.ScaleLinear<number, number>
    series: PointValue[]
    isPrime?: boolean
}

export class LinePlot {
    series: PointValue[]
    xScale: d3.ScaleLinear<number, number>
    yScale: d3.ScaleLinear<number, number>
    color: string
    circleElem: WeyaElement
    line: d3.Line<PointValue>
    bisect: d3.Bisector<number, number>
    isPrime: boolean

    constructor(opt: LinePlotOptions) {
        this.series = opt.series
        this.xScale = opt.xScale
        this.yScale = opt.yScale
        this.color = opt.color
        this.isPrime = opt.isPrime

        this.bisect = d3.bisector(function (d: PointValue) {
            return d.step
        }).left

        this.line = d3.line()
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
            if (this.isPrime) {
                $('path.smoothed-line.dropshadow',
                    {
                        fill: 'none',
                        stroke: this.color,
                        d: this.line(this.series) as string
                    })
            } else {
                $('path.unsmoothed-line',
                    {
                        fill: 'none',
                        stroke: this.color,
                        d: this.line(this.series) as string
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
            this.circleElem.setAttribute("cy", `${this.yScale(this.series[idx].value)}`)
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
    line: d3.Line<PointValue>
    fill: string
    dFill: string

    constructor(opt: LineFillOptions) {
        this.series = opt.series
        this.xScale = opt.xScale
        this.yScale = opt.yScale
        this.color = opt.color
        this.colorIdx = opt.colorIdx
        this.chartId = opt.chartId

        this.line = d3.line()
            .curve(d3.curveMonotoneX)
            .x((d, i) => {
                return this.xScale(d.step)
            })
            .y((d) => {
                return this.yScale(d.value)
            })

        let d = this.line(this.series) as string
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
