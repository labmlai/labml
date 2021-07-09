import d3 from "../../../d3"
import {WeyaElement, WeyaElementFunction} from '../../../../../lib/weya/weya'
import {ChartOptions} from '../types'
import {SeriesModel} from "../../../models/run"
import {defaultSeriesToPlot, getExtent, getLogScale, getScale} from "../utils"
import {CompareLinePlot, LineFill} from "./plot"
import {BottomAxis, RightAxis} from "../axis"
import {formatStep} from "../../../utils/value"
import {DropShadow, LineGradients} from "../chart_gradients"
import ChartColors from "../chart_colors"
import {getWindowDimensions} from '../../../utils/window_dimentions'

const LABEL_HEIGHT = 10

interface CompareLineChartOptions extends ChartOptions {
    baseSeries: SeriesModel[]
    currentPlotIdx: number[]
    basePlotIdx: number[]
    chartType: string
    onCursorMove?: ((cursorStep?: number | null) => void)[]
    isCursorMoveOpt?: boolean
    isDivergent?: boolean
}

export class CompareLineChart {
    currentSeries: SeriesModel[]
    baseSeries: SeriesModel[]
    currentPlotIdx: number[]
    basePlotIdx: number[]
    currentPlot: SeriesModel[] = []
    basePlot: SeriesModel[] = []
    filteredCurrentPlotIdx: number[] = []
    filteredBasePlotIdx: number[] = []
    chartType: string
    chartWidth: number
    chartHeight: number
    margin: number
    axisSize: number
    xScale: d3.ScaleLinear<number, number>
    yScale: d3.ScaleLinear<number, number>
    svgElem: WeyaElement
    stepElement: WeyaElement
    linePlots: CompareLinePlot[] = []
    onCursorMove?: ((cursorStep?: number | null) => void)[]
    isCursorMoveOpt?: boolean
    chartColors: ChartColors
    isDivergent: boolean
    private svgBoundingClientRect: DOMRect

    constructor(opt: CompareLineChartOptions) {
        this.currentSeries = opt.series
        this.baseSeries = opt.baseSeries
        this.chartType = opt.chartType
        this.currentPlotIdx = opt.currentPlotIdx
        this.basePlotIdx = opt.basePlotIdx
        this.onCursorMove = opt.onCursorMove ? opt.onCursorMove : []
        this.isCursorMoveOpt = opt.isCursorMoveOpt

        this.axisSize = 30
        let windowWidth = opt.width
        let windowHeight = getWindowDimensions().height
        this.margin = Math.floor(windowWidth / 64)
        this.chartWidth = windowWidth - 2 * this.margin - this.axisSize
        this.chartHeight = Math.round(Math.min(this.chartWidth, windowHeight) / 2)

        if (this.currentPlotIdx.length === 0) {
            this.currentPlotIdx = defaultSeriesToPlot(this.currentSeries)
        }
        if (this.basePlotIdx.length === 0) {
            this.basePlotIdx = defaultSeriesToPlot(this.baseSeries)
        }

        for (let i = 0; i < this.currentPlotIdx.length; i++) {
            if (this.currentPlotIdx[i] >= 0) {
                this.filteredCurrentPlotIdx.push(i)
                this.currentPlot.push(this.currentSeries[i])
            }
        }
        for (let i = 0; i < this.basePlotIdx.length; i++) {
            if (this.basePlotIdx[i] >= 0) {
                this.filteredBasePlotIdx.push(i)
                this.basePlot.push(this.baseSeries[i])
            }
        }
        if (this.currentPlotIdx.length > 0 && Math.max(...this.currentPlotIdx) < 0) {
            this.currentPlot = [this.currentSeries[0]]
            this.filteredCurrentPlotIdx = [0]
        }
        if (this.basePlotIdx.length > 0 && Math.max(...this.basePlotIdx) < 0) {
            this.basePlot = [this.baseSeries[0]]
            this.filteredBasePlotIdx = [0]
        }

        const stepExtent = getExtent(this.currentSeries.concat(this.baseSeries).map(s => s.series), d => d.step)
        this.xScale = getScale(stepExtent, this.chartWidth, false)

        this.chartColors = new ChartColors({nColors: this.currentSeries.length, isDivergent: opt.isDivergent})
    }

    chartId = `chart_${Math.round(Math.random() * 1e9)}`

    changeScale() {
        let plotSeries = this.currentPlot.concat(this.basePlot).map(s => s.series)

        if (this.chartType === 'log') {
            this.yScale = getLogScale(getExtent(plotSeries, d => d.value, false, true), -this.chartHeight)
        } else {
            this.yScale = getScale(getExtent(plotSeries, d => d.value, false), -this.chartHeight)
        }
    }

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
        let cursorStep: number = null

        if (this.svgBoundingClientRect == null) {
            return
        }

        if (clientX) {
            let currentX = this.xScale.invert(clientX - this.svgBoundingClientRect.left - this.margin)
            if (currentX > 0) {
                cursorStep = currentX
            }
        }

        this.renderStep(cursorStep)
        for (let linePlot of this.linePlots) {
            linePlot.renderCursorCircle(cursorStep)
        }
        for (let func of this.onCursorMove) {
            func(cursorStep)
        }
    }

    renderStep(cursorStep: number) {
        this.stepElement.textContent = `Step : ${formatStep(cursorStep)}`
    }

    render($: WeyaElementFunction) {
        this.changeScale()

        if (this.currentSeries.concat(this.baseSeries).length === 0) {
            $('div', '')
        } else {
            $('div', $ => {
                $('div', $ => {
                        this.svgElem = $('svg', '#chart',
                            {
                                height: LABEL_HEIGHT + 2 * this.margin + this.axisSize + this.chartHeight,
                                width: 2 * this.margin + this.axisSize + this.chartWidth,
                            }, $ => {
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
                                        if (this.currentPlot.length < 3) {
                                            this.currentPlot.map((s, i) => {
                                                new LineFill({
                                                    series: s.series,
                                                    xScale: this.xScale,
                                                    yScale: this.yScale,
                                                    color: this.chartColors.getColor(this.filteredCurrentPlotIdx[i]),
                                                    colorIdx: this.filteredCurrentPlotIdx[i],
                                                    chartId: this.chartId
                                                }).render($)
                                            })
                                        }
                                        this.currentPlot.map((s, i) => {
                                            let linePlot = new CompareLinePlot({
                                                series: s.series,
                                                xScale: this.xScale,
                                                yScale: this.yScale,
                                                color: this.chartColors.getColor(this.filteredCurrentPlotIdx[i])
                                            })
                                            this.linePlots.push(linePlot)
                                            linePlot.render($)
                                        })
                                        this.basePlot.map((s, i) => {
                                            let linePlot = new CompareLinePlot({
                                                series: s.series,
                                                xScale: this.xScale,
                                                yScale: this.yScale,
                                                color: this.chartColors.getColor(this.filteredBasePlotIdx[i]),
                                                isDotted: true
                                            })
                                            this.linePlots.push(linePlot)
                                            linePlot.render($)
                                        })
                                    })
                                $('g.bottom-axis',
                                    {
                                        transform: `translate(${this.margin}, ${this.margin + this.chartHeight + LABEL_HEIGHT})`
                                    }, $ => {
                                        new BottomAxis({chartId: this.chartId, scale: this.xScale}).render($)
                                    })
                                $('g.right-axis',
                                    {
                                        transform: `translate(${this.margin + this.chartWidth}, ${this.margin + this.chartHeight})`
                                    }, $ => {
                                        new RightAxis({chartId: this.chartId, scale: this.yScale}).render($)
                                    })
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
