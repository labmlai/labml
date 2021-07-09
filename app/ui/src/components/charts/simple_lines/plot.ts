import d3 from "../../../d3"
import {WeyaElementFunction} from '../../../../../lib/weya/weya'
import {FillOptions, PlotOptions} from '../types'

interface SimpleLinePlotOptions extends PlotOptions {
    xScale: d3.ScaleLinear<number, number>
    series: number[]
}

export class SimpleLinePlot {
    series: number[]
    xScale: d3.ScaleLinear<number, number>
    yScale: d3.ScaleLinear<number, number>
    color: string
    smoothedLine: d3.Line<number>

    constructor(opt: SimpleLinePlotOptions) {
        this.series = opt.series
        this.xScale = opt.xScale
        this.yScale = opt.yScale
        this.color = opt.color

        this.smoothedLine = d3.line()
            .curve(d3.curveMonotoneX)
            .x((d, i) => {
                return this.xScale(i)
            })
            .y((d) => {
                return this.yScale(d)
            })
    }


    render($: WeyaElementFunction) {
        $('g', $ => {
            $('path.smoothed-line.dropshadow',
                {
                    fill: 'none',
                    stroke: this.color,
                    d: this.smoothedLine(this.series) as string
                })
        })
    }
}

interface SimpleLineFillOptions extends FillOptions {
    xScale: d3.ScaleLinear<number, number>
    series: number[]
    chartId: string
}

export class SimpleLineFill {
    series: number[]
    chartId: string
    xScale: d3.ScaleLinear<number, number>
    yScale: d3.ScaleLinear<number, number>
    color: string
    colorIdx: number
    smoothedLine
    dFill: string

    constructor(opt: SimpleLineFillOptions) {
        this.series = opt.series
        this.xScale = opt.xScale
        this.yScale = opt.yScale
        this.color = opt.color
        this.colorIdx = opt.colorIdx
        this.chartId = opt.chartId

        this.smoothedLine = d3.line()
            .curve(d3.curveMonotoneX)
            .x((d, i) => {
                return this.xScale(i)
            })
            .y((d) => {
                return this.yScale(d)
            })

        let d = this.smoothedLine(this.series) as string
        this.dFill = `M${this.xScale(0)},0L` + d.substr(1) + `L${this.xScale(this.series.length - 1)},0`
    }

    render($: WeyaElementFunction) {
        $('g', $ => {
            $('path.line-fill',
                {
                    fill: this.color,
                    stroke: 'none',
                    style: {fill: `url(#gradient-${this.colorIdx}-${this.chartId}`},
                    d: this.dFill
                })
        })
    }
}
