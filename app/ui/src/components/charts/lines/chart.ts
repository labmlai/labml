import d3 from "../../../d3"
import {WeyaElement, WeyaElementFunction} from '../../../../../lib/weya/weya'
import {ChartOptions} from '../types'
import {PointValue, SeriesModel} from "../../../models/run"
import {
    getExtent,
    getLogScale,
    getScale,
    getSmoothWindow,
    smoothSeries,
    trimSteps, trimStepsOfPoints
} from "../utils"
import {LineFill, LinePlot} from "./plot"
import {BottomAxis, RightAxis} from "../axis"
import {formatStep} from "../../../utils/value"
import {DropShadow, LineGradients} from "../chart_gradients"
import ChartColors from "../chart_colors"
import {getWindowDimensions} from '../../../utils/window_dimentions'

const LABEL_HEIGHT = 10

interface LineChartOptions extends ChartOptions {
    baseSeries?: SeriesModel[]
    basePlotIdx?: number[]
    // series is defined in chart options - used as current series
    plotIndex: number[]
    onSelect?: (i: number) => void
    chartType: string
    onCursorMove?: ((cursorStep?: number | null) => void)[]
    isCursorMoveOpt?: boolean
    isDivergent?: boolean
    stepRange: number[]
    focusSmoothed: boolean
    smoothValue: number
}

export class LineChart {
    private readonly currentSeries: SeriesModel[]
    private readonly currentPlotIndex: number[]
    private readonly baseSeries: SeriesModel[]
    private readonly basePlotIndex: number[]
    private readonly filteredBaseSeries: SeriesModel[]
    private readonly filteredCurrentSeries: SeriesModel[]
    private readonly chartType: string
    private readonly chartWidth: number
    private readonly chartHeight: number
    private readonly margin: number
    private readonly axisSize: number
    private xScale: d3.ScaleLinear<number, number>
    private yScale: d3.ScaleLinear<number, number>
    private svgElem: WeyaElement
    private stepElement: WeyaElement
    private readonly linePlots: LinePlot[] = []
    private readonly onCursorMove?: ((cursorStep?: number | null) => void)[]
    private readonly isCursorMoveOpt?: boolean
    private readonly chartColors: ChartColors
    private svgBoundingClientRect: DOMRect
    private readonly  uniqueItems: Map<string, number>
    private readonly focusSmoothed: boolean

    private readonly currentSmoothedSeriesPoints: PointValue[][]
    private readonly baseSmoothedSeriesPoints: PointValue[][]

    constructor(opt: LineChartOptions) {
        this.currentSeries = opt.series
        this.currentPlotIndex = opt.plotIndex
        this.baseSeries = opt.baseSeries ?? []
        this.basePlotIndex = opt.basePlotIdx ?? []
        this.chartType = opt.chartType
        this.onCursorMove = opt.onCursorMove ? opt.onCursorMove : []
        this.isCursorMoveOpt = opt.isCursorMoveOpt
        this.focusSmoothed = opt.focusSmoothed

        this.uniqueItems = new Map<string, number>()
        this.axisSize = 30
        let windowWidth = opt.width
        let windowHeight = getWindowDimensions().height
        this.margin = Math.floor(windowWidth / 64)
        this.chartWidth = windowWidth - 2 * this.margin - this.axisSize
        this.chartHeight = Math.round(Math.min(this.chartWidth, windowHeight) / 2)

        let smoothWindow = getSmoothWindow(this.currentSeries, this.baseSeries, opt.smoothValue)

        this.filteredBaseSeries = this.baseSeries.filter((_, i) => this.basePlotIndex[i] == 1)
        this.filteredCurrentSeries = this.currentSeries.filter((_, i) => this.currentPlotIndex[i] == 1)

        this.currentSmoothedSeriesPoints = this.filteredCurrentSeries.map(s => {
          return smoothSeries(s.series, smoothWindow)
        })
        this.baseSmoothedSeriesPoints = this.filteredBaseSeries.map(s => {
          return smoothSeries(s.series, smoothWindow)
        })

        this.filteredBaseSeries = trimSteps(this.filteredBaseSeries, opt.stepRange[0], opt.stepRange[1], smoothWindow)
        this.filteredCurrentSeries = trimSteps(this.filteredCurrentSeries, opt.stepRange[0], opt.stepRange[1], smoothWindow)

        this.currentSmoothedSeriesPoints = trimStepsOfPoints(this.currentSmoothedSeriesPoints, opt.stepRange[0], opt.stepRange[1], smoothWindow)
        this.baseSmoothedSeriesPoints = trimStepsOfPoints(this.baseSmoothedSeriesPoints, opt.stepRange[0], opt.stepRange[1], smoothWindow)

        const stepExtent = getExtent(this.filteredBaseSeries.concat(this.filteredCurrentSeries).map(s => s.series), d => d.step, false, true)
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
        let plotSeries = this.filteredBaseSeries.concat(this.filteredCurrentSeries).map(s => s.series)
        if (plotSeries.length == 0) {
            this.yScale = d3.scaleLinear()
                .domain([0, 0])
                .range([0, 0]);
            return
        }
        if (this.chartType === 'log') {
            this.yScale = getLogScale(getExtent(plotSeries, d => this.focusSmoothed ? d.smoothed : d.value, false, true), -this.chartHeight)
        } else {
            this.yScale = getScale(getExtent(plotSeries, d => this.focusSmoothed ? d.smoothed : d.value, false), -this.chartHeight)
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
            if (clientX > this.svgBoundingClientRect.right) {
                clientX = this.svgBoundingClientRect.right
            }

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

        $('div', $ => {
            if (this.filteredBaseSeries.length + this.filteredCurrentSeries.length == 0) {
                $('div', '.chart-overlay', $ => {
                    $('span', '.text', 'No Metric Selected')
                })
            }
            $('div', $ => {
                    this.stepElement = $('h6', '.text-center.selected-step', '')
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
                                    if (this.filteredCurrentSeries.length < 3 && this.filteredBaseSeries.length == 0) {
                                        this.filteredCurrentSeries.map((s, i) => {
                                            if (this.currentSmoothedSeriesPoints[i].length == 0) {
                                                return
                                            }
                                            new LineFill({
                                                series: this.currentSmoothedSeriesPoints[i],
                                                xScale: this.xScale,
                                                yScale: this.yScale,
                                                color: document.body.classList.contains("light")
                                                    ? this.chartColors.getSecondColor(this.uniqueItems.get(s.name))
                                                    : this.chartColors.getColor(this.uniqueItems.get(s.name)),
                                                colorIdx: this.uniqueItems.get(s.name),
                                                chartId: this.chartId
                                            }).render($)
                                        })
                                    }
                                    this.filteredCurrentSeries.map((s, i) => {
                                        let linePlot = new LinePlot({
                                            series: s.series,
                                            xScale: this.xScale,
                                            yScale: this.yScale,
                                            color: document.body.classList.contains("light")
                                                    ? this.chartColors.getSecondColor(this.uniqueItems.get(s.name))
                                                    : this.chartColors.getColor(this.uniqueItems.get(s.name)),
                                            renderHorizontalLine: true,
                                            smoothFocused: this.focusSmoothed,
                                            smoothedSeries: this.currentSmoothedSeriesPoints[i],
                                        })
                                        this.linePlots.push(linePlot)
                                        linePlot.render($)
                                    })

                                    this.filteredBaseSeries.map((s, i) => {
                                        let linePlot = new LinePlot({
                                            series: s.series,
                                            xScale: this.xScale,
                                            yScale: this.yScale,
                                            color: document.body.classList.contains("light")
                                                    ? this.chartColors.getColor(this.uniqueItems.get(s.name))
                                                    : this.chartColors.getSecondColor(this.uniqueItems.get(s.name)),
                                            isBase: true,
                                            renderHorizontalLine: true,
                                            smoothFocused: this.focusSmoothed,
                                            smoothedSeries: this.baseSmoothedSeriesPoints[i],
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
