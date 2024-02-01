import d3 from "../../../d3"
import {WeyaElement, WeyaElementFunction} from '../../../../../lib/weya/weya'
import {ChartOptions} from '../types'
import {SeriesModel} from "../../../models/run"
import {fillPlotPreferences, getExtent, getScale, getTimeScale, toDate} from "../utils"
import {TimeSeriesFill, TimeSeriesPlot} from './plot'
import {BottomTimeAxis, RightAxis} from "../axis"
import {DefaultLineGradient, DropShadow, LineGradients} from "../chart_gradients"
import ChartColors from "../chart_colors"
import {getWindowDimensions} from '../../../utils/window_dimentions'
import {formatDateTime} from "../../../utils/time"

const LABEL_HEIGHT = 10

interface LineChartOptions extends ChartOptions {
    plotIdx: number[]
    onSelect?: (i: number) => void
    onCursorMove?: ((cursorStep?: Date | null) => void)[]
    isCursorMoveOpt?: boolean
    isDivergent?: boolean
}

export class SingleScaleLineChart {
    series: SeriesModel[]
    plotIdx: number[]
    plot: SeriesModel[] = []
    filteredPlotIdx: number[] = []
    chartWidth: number
    chartHeight: number
    margin: number
    axisSize: number
    labels: string[] = []
    xScale: d3.ScaleTime<number, number>
    svgElem: WeyaElement
    stepElement: WeyaElement
    timeSeriesPlots: TimeSeriesPlot[] = []
    onCursorMove?: ((cursorStep?: Date | null) => void)[]
    isCursorMoveOpt?: boolean
    chartColors: ChartColors
    isDivergent: boolean
    private yScales: { [name: string]: d3.ScaleLinear<number, number> } = {}
    private svgBoundingClientRect: DOMRect

    constructor(opt: LineChartOptions) {
        this.series = opt.series
        this.plotIdx = opt.plotIdx
        this.onCursorMove = opt.onCursorMove ? opt.onCursorMove : []
        this.isCursorMoveOpt = opt.isCursorMoveOpt

        this.axisSize = 30
        let windowWidth = opt.width
        let windowHeight = getWindowDimensions().height
        this.margin = Math.floor(windowWidth / 64)
        this.chartWidth = windowWidth - 2 * this.margin - this.axisSize
        this.chartHeight = Math.round(Math.min(this.chartWidth, windowHeight) / 2)

        if (this.plotIdx.length === 0) {
            this.plotIdx = fillPlotPreferences(this.series)
        }

        for (let i = 0; i < this.plotIdx.length; i++) {
            if (this.plotIdx[i] >= 0) {
                this.filteredPlotIdx.push(i)
                this.plot.push(this.series[i])
            }
        }
        if (this.plotIdx.length > 0 && Math.max(...this.plotIdx) < 0) {
            this.plot = [this.series[0]]
            this.filteredPlotIdx = [0]
        }

        for (let s of this.plot) {
            this.yScales[s.name] = getScale(getExtent([s.series], d => d.value, false), -this.chartHeight)
        }

        const stepExtent = getExtent(this.series.map(s => s.series), d => d.step)
        this.xScale = getTimeScale([toDate(stepExtent[0]), toDate(stepExtent[1])], this.chartWidth)

        this.chartColors = new ChartColors({nColors: this.series.length, isDivergent: opt.isDivergent})
    }

    chartId = `chart_${Math.round(Math.random() * 1e9)}`

    onTouchStart = (ev: TouchEvent) => {
        if (ev.touches.length !== 1) return
        this.updateCursorStep(ev.touches[0].clientX)
    }

    onTouchMove = (ev: TouchEvent) => {
        if (ev.touches.length !== 1) return
        this.updateCursorStep(ev.touches[0].clientX)
    }

    onTouchEnd = (ev: TouchEvent) => {
        if (ev.touches.length !== 1) return
        this.updateCursorStep(ev.touches[0].clientX)
    }

    onMouseDown = (ev: MouseEvent) => {
        this.updateCursorStep(ev.clientX)
    }

    onMouseUp = (ev: MouseEvent) => {
        this.updateCursorStep(ev.clientX)
    }

    onMouseMove = (ev: MouseEvent) => {
        this.updateCursorStep(ev.clientX)
    }

    updateCursorStep(clientX: number) {
        let cursorStep: Date = null

        if (this.svgBoundingClientRect == null) {
            return
        }

        if (clientX) {
            let currentX = this.xScale.invert(clientX - this.svgBoundingClientRect.left - this.margin)
            cursorStep = currentX
        }

        this.renderStep(cursorStep)
        for (let timeSeriesPlot of this.timeSeriesPlots) {
            timeSeriesPlot.renderCursorCircle(cursorStep)
        }
        for (let func of this.onCursorMove) {
            func(cursorStep)
        }
    }

    renderStep(cursorStep: Date) {
        this.stepElement.textContent = `${formatDateTime(cursorStep)}`
    }

    render($: WeyaElementFunction) {
        if (this.series.length === 0) {
            $('div', '')
        } else {
            $('div', $ => {
                $('div', $ => {
                        this.svgElem = $('svg', '#chart',
                            {
                                height: LABEL_HEIGHT + 2 * this.margin + this.axisSize + this.chartHeight,
                                width: 2 * this.margin + this.axisSize + this.chartWidth,
                            }, $ => {
                                new DefaultLineGradient().render($)
                                new DropShadow().render($)
                                new LineGradients({chartColors: this.chartColors, chartId: this.chartId}).render($)
                                $('g', {}, $ => {
                                    this.stepElement = $('text', '.selected-step',
                                        {transform: `translate(${(2 * this.margin + this.axisSize + this.chartWidth) / 2},${LABEL_HEIGHT})`})
                                })
                                $('g',
                                    {
                                        transform: `translate(${this.margin}, ${this.margin + this.chartHeight})`
                                    }, $ => {
                                        if (this.plot.length < 3) {
                                            this.plot.map((s, i) => {
                                                new TimeSeriesFill({
                                                    series: s.series,
                                                    xScale: this.xScale,
                                                    yScale: this.yScales[s.name],
                                                    color: this.chartColors.getColor(this.filteredPlotIdx[i]),
                                                    colorIdx: this.filteredPlotIdx[i],
                                                    chartId: this.chartId
                                                }).render($)
                                            })
                                        }
                                        this.plot.map((s, i) => {
                                            let linePlot = new TimeSeriesPlot({
                                                series: s.series,
                                                xScale: this.xScale,
                                                yScale: this.yScales[s.name],
                                                color: this.chartColors.getColor(this.filteredPlotIdx[i])
                                            })
                                            this.timeSeriesPlots.push(linePlot)
                                            linePlot.render($)
                                        })
                                    })
                                $('g.bottom-axis',
                                    {
                                        transform: `translate(${this.margin}, ${this.margin + this.chartHeight})`
                                    },
                                    $ => {
                                        new BottomTimeAxis({chartId: this.chartId, scale: this.xScale}).render($)
                                    })
                                if (this.plot.length == 1) {
                                    $('g.right-axis',
                                        {
                                            transform: `translate(${this.margin + this.chartWidth}, ${this.margin + this.chartHeight})`
                                        }, $ => {
                                            new RightAxis({
                                                chartId: this.chartId,
                                                scale: this.yScales[this.plot[0].name],
                                                specifier: '.1s',
                                                numTicks: 3
                                            }).render($)
                                        })
                                }
                            })
                    }
                )
            })

            if (this.isCursorMoveOpt) {
                this.svgElem.addEventListener('touchstart', this.onTouchStart)
                this.svgElem.addEventListener('touchmove', this.onTouchMove)
                this.svgElem.addEventListener('touchend', this.onTouchEnd)
                this.svgElem.addEventListener('mouseup', this.onMouseUp)
                this.svgElem.addEventListener('mousemove', this.onMouseMove)
                this.svgElem.addEventListener('mousedown', this.onMouseDown)
            }

            this.svgBoundingClientRect = null

            window.requestAnimationFrame(() => {
                this.svgBoundingClientRect = this.svgElem.getBoundingClientRect()
            })
        }
    }
}
