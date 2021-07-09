import {WeyaElementFunction} from '../../../../../lib/weya/weya'
import {ChartOptions} from '../types'
import {SeriesModel} from "../../../models/run"
import {getScale} from "../utils"
import d3 from "../../../d3"
import {SimpleLineFill, SimpleLinePlot} from "./plot"
import {RightAxis} from "../axis"
import {Labels} from "../labels"
import {DropShadow, LineGradients} from "../chart_gradients"
import ChartColors from "../chart_colors"


export class SimpleLinesChart {
    series: SeriesModel[]
    chartWidth: number
    chartHeight: number
    margin: number
    axisSize: number
    labels: string[] = []
    xScale: d3.ScaleLinear<number, number>
    yScale: d3.ScaleLinear<number, number>
    chartColors: ChartColors

    constructor(opt: ChartOptions) {
        this.series = opt.series

        this.axisSize = 30
        let windowWidth = opt.width
        this.margin = Math.floor(windowWidth / 64)
        this.chartWidth = windowWidth - 2 * this.margin - this.axisSize
        this.chartHeight = Math.round(this.chartWidth / 4)

        let plot: number[] = []
        for (let s of this.series) {
            plot.push(...s.value)
            this.labels.push(s.name)
        }

        this.xScale = getScale([0, this.series[0].value.length - 1], this.chartWidth, false)
        this.yScale = getScale([Math.min(...plot), Math.max(...plot)], -this.chartHeight)

        this.chartColors = new ChartColors({nColors: this.series.length})
    }

    chartId = `chart_${Math.round(Math.random() * 1e9)}`

    render($: WeyaElementFunction) {
        if (this.series.length === 0) {
            $('div', '')
        } else {
            $('div', $ => {
                $('svg',
                    {
                        id: 'chart',
                        height: 2 * this.margin + this.chartHeight,
                        width: 2 * this.margin + this.axisSize + this.chartWidth
                    },
                    $ => {
                        new DropShadow().render($)
                        new LineGradients({chartColors: this.chartColors, chartId: this.chartId}).render($)
                        $('g',
                            {
                                transform: `translate(${this.margin}, ${this.margin + this.chartHeight})`
                            },
                            $ => {
                                $('g', $ => {
                                    this.series.map((s, i) => {
                                        new SimpleLineFill({
                                            series: s.value,
                                            xScale: this.xScale,
                                            yScale: this.yScale,
                                            color: this.chartColors.getColor(i),
                                            colorIdx: i,
                                            chartId: this.chartId
                                        }).render($)
                                    })
                                    this.series.map((s, i) => {
                                        new SimpleLinePlot({
                                            series: s.value,
                                            xScale: this.xScale,
                                            yScale: this.yScale,
                                            color: this.chartColors.getColor(i)
                                        }).render($)
                                    })
                                })
                            })
                        $('g.right-axis',
                            {
                                transform: `translate(${this.margin + this.chartWidth}, ${this.margin + this.chartHeight})`
                            },
                            $ => {
                                new RightAxis({chartId: this.chartId, scale: this.yScale}).render($)
                            })
                    })
                new Labels({labels: this.labels, chartColors: this.chartColors}).render($)
            })
        }
    }
}