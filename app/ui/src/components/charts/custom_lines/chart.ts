import d3 from "../../../d3"
import {WeyaElement, WeyaElementFunction} from '../../../../../lib/weya/weya'
import {SeriesModel} from "../../../models/run"
import {defaultSeriesToPlot, getExtent, getScale, toPointValue} from "../utils"
import {LineFill, LinePlot} from "./plot"
import {BottomAxis, RightAxis} from "../axis"
import {formatStep} from "../../../utils/value"
import {DropShadow, LineGradients} from "../chart_gradients"
import ChartColors from "../chart_colors"
import {getWindowDimensions} from '../../../utils/window_dimentions'

const LABEL_HEIGHT = 10

interface LineChartOptions {
    width: number
    series: SeriesModel[],
    plotIdx: number[]
    onSelect?: (i: number) => void
    onCursorMove?: ((cursorStep?: number | null) => void)[]
    isCursorMoveOpt?: boolean
}

export class CustomLineChart {
    series: SeriesModel[]
    plotIdx: number[]
    filteredPlots: SeriesModel[] = []
    subPlots: SeriesModel[] = []
    filteredPlotIdx: number[] = []
    chartWidth: number
    chartHeight: number
    margin: number
    axisSize: number
    labels: string[] = []
    xScale: d3.ScaleLinear<number, number>
    yScale: d3.ScaleLinear<number, number>
    svgElem: WeyaElement
    stepElement: WeyaElement
    linePlots: LinePlot[] = []
    onCursorMove?: ((cursorStep?: number | null) => void)[]
    isCursorMoveOpt?: boolean
    chartColors: ChartColors
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
            this.plotIdx = defaultSeriesToPlot(this.series)
        }

        for (let i = 0; i < this.plotIdx.length; i++) {
            if (this.plotIdx[i] >= 0) {
                let s = this.series[i]
                this.filteredPlotIdx.push(i)
                this.filteredPlots.push(s)
                if (s.sub) {
                    s.sub.series = toPointValue(s.sub)
                    this.subPlots.push(s.sub)
                }
            }
        }

        if (this.plotIdx.length > 0 && Math.max(...this.plotIdx) < 0) {
            this.filteredPlots = [this.series[0]]
            this.filteredPlotIdx = [0]
        }

        let plots: SeriesModel[] = this.filteredPlots.concat(this.subPlots)

        const stepExtent = getExtent(plots.map(s => s.series), d => d.step)
        this.xScale = getScale(stepExtent, this.chartWidth, false)
        this.yScale = getScale(getExtent(plots.map(s => s.series), d => d.value, false), -this.chartHeight)

        this.chartColors = new ChartColors({nColors: this.series.length, isDivergent: true})
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
                                        if (this.filteredPlots.length < 3) {
                                            this.filteredPlots.map((s, i) => {
                                                new LineFill({
                                                    series: s.series,
                                                    xScale: this.xScale,
                                                    yScale: this.yScale,
                                                    color: this.chartColors.getColor(this.filteredPlotIdx[i]),
                                                    colorIdx: this.filteredPlotIdx[i],
                                                    chartId: this.chartId
                                                }).render($)
                                            })
                                        }
                                        this.filteredPlots.map((s, i) => {
                                            let lineColor = this.chartColors.getColor(this.filteredPlotIdx[i])
                                            let linePlot = new LinePlot({
                                                series: s.series,
                                                xScale: this.xScale,
                                                yScale: this.yScale,
                                                color: lineColor,
                                                isPrime: true
                                            })
                                            this.linePlots.push(linePlot)
                                            linePlot.render($)
                                            if (s.sub) {
                                                new LinePlot({
                                                    series: s.sub.series,
                                                    xScale: this.xScale,
                                                    yScale: this.yScale,
                                                    color: lineColor,
                                                }).render($)
                                            }
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
