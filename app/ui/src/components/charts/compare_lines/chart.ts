import d3 from "../../../d3"
import {WeyaElement, WeyaElementFunction} from '../../../../../lib/weya/weya'
import {ChartOptions} from '../types'
import {SeriesModel} from "../../../models/run"
import {defaultSeriesToPlot, getExtent, getLogScale, getScale} from "../utils"
import {LineFill, LinePlot} from "./plot"
import {BottomAxis, RightAxis} from "../axis"
import {formatStep} from "../../../utils/value"
import {DropShadow, LineGradients} from "../chart_gradients"
import ChartColors from "../chart_colors"
import {getWindowDimensions} from '../../../utils/window_dimentions'

const LABEL_HEIGHT = 10

interface LineChartOptions extends ChartOptions {
    baseSeries: SeriesModel[]
    basePlotIdx: number[]
    // series is defined in chart options - used as current series
    currentPlotIndex: number[]
    onSelect?: (i: number) => void
    chartType: string
    onCursorMove?: ((cursorStep?: number | null) => void)[]
    isCursorMoveOpt?: boolean
    isDivergent?: boolean
}

export class LineChart {
    private readonly currentSeries: SeriesModel[]
    private readonly currentPlotIndex: number[]
    private readonly baseSeries: SeriesModel[]
    private readonly basePlotIndex: number[]
    chartType: string
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
    private chartColors: ChartColors
    isDivergent: boolean
    private svgBoundingClientRect: DOMRect
    private uniqueItems: Map<string, number>

    constructor(opt: LineChartOptions) {
        this.currentSeries = opt.series
        this.currentPlotIndex = opt.currentPlotIndex
        this.baseSeries = opt.baseSeries
        this.basePlotIndex = opt.basePlotIdx
        this.chartType = opt.chartType
        this.onCursorMove = opt.onCursorMove ? opt.onCursorMove : []
        this.isCursorMoveOpt = opt.isCursorMoveOpt
        this.uniqueItems = new Map<string, number>()
        this.axisSize = 30
        let windowWidth = opt.width
        let windowHeight = getWindowDimensions().height
        this.margin = Math.floor(windowWidth / 64)
        this.chartWidth = windowWidth - 2 * this.margin - this.axisSize
        this.chartHeight = Math.round(Math.min(this.chartWidth, windowHeight) / 2)

        if (this.currentPlotIndex.length == 0) {
            this.currentPlotIndex = defaultSeriesToPlot(this.currentSeries)
        }
        if (this.basePlotIndex.length == 0) {
            this.basePlotIndex = defaultSeriesToPlot(this.baseSeries)
        }

        const stepExtent = getExtent(this.currentSeries.concat(this.baseSeries).map(s => s.series), d => d.step)
        this.xScale = getScale(stepExtent, this.chartWidth, false)

        let idx: number = 0
        for (let s of this.currentSeries.concat(this.baseSeries)) {
            if (!this.uniqueItems.has(s.name)) {
                this.uniqueItems.set(s.name, idx++)
            }
        }

        this.chartColors = new ChartColors({
            nColors: this.uniqueItems.size,
            secondNColors: this.uniqueItems.size,
            isDivergent: opt.isDivergent})
    }

    chartId = `chart_${Math.round(Math.random() * 1e9)}`

    changeScale() {
        let plotSeries = this.currentSeries.flatMap((s, i) => this.currentPlotIndex[i]<0 ? [] : [s.series])
            .concat(this.baseSeries.flatMap((s, i) => this.basePlotIndex[i]<0 ? [] : [s.series]))
        if (plotSeries.length == 0) {
            return
        }
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
            linePlot.renderIndicators(cursorStep)
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
        let filteredBaseSeriesLength = this.baseSeries.filter((_, i) => this.basePlotIndex[i] >= 0).length
        let filteredCurrentSeriesLength = this.currentSeries.filter((_, i) => this.currentPlotIndex[i] >= 0).length

        if (filteredBaseSeriesLength + filteredCurrentSeriesLength === 0) {
            $('div', '')
        } else {
            $('div', $ => {
                $('div', $ => {
                        // this.stepElement = $('h6', '.text-center.selected-step', '')
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
                                        this.baseSeries.map((s, i) => {
                                            if (this.basePlotIndex[i] < 0) {
                                                    return;
                                                }
                                            let linePlot = new LinePlot({
                                                series: s.series,
                                                xScale: this.xScale,
                                                yScale: this.yScale,
                                                color: document.body.classList.contains("light")
                                                        ? this.chartColors.getColor(this.uniqueItems.get(s.name))
                                                        : this.chartColors.getSecondColor(this.uniqueItems.get(s.name)),
                                                isBase: true,
                                                renderHorizontalLine: true
                                            })
                                            this.linePlots.push(linePlot)
                                            linePlot.render($)
                                        })
                                        this.currentSeries.map((s, i) => {
                                            if (this.currentPlotIndex[i] < 0) {
                                                    return;
                                                }
                                            let linePlot = new LinePlot({
                                                series: s.series,
                                                xScale: this.xScale,
                                                yScale: this.yScale,
                                                color: document.body.classList.contains("light")
                                                        ? this.chartColors.getSecondColor(this.uniqueItems.get(s.name))
                                                        : this.chartColors.getColor(this.uniqueItems.get(s.name)),
                                                renderHorizontalLine: true
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
