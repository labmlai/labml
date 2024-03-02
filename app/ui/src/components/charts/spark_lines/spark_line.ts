import d3 from "../../../d3"
import {WeyaElementFunction} from '../../../../../lib/weya/weya'
import {PointValue} from "../../../models/run"
import {getBaseColor} from "../constants"
import {getExtent, getScale, getSelectedIdx} from "../utils"
import {formatFixed} from "../../../utils/value"
import {LineFill, LinePlot} from '../lines/plot'

export interface SparkLineOptions {
    name: string
    series: PointValue[]
    width: number
    stepExtent: [number, number]
    selected: number
    onClick?: () => void
    isMouseMoveOpt?: boolean
    color: string
    isBase?: boolean
    smoothedValues: number[]
}

export class SparkLine {
    series: PointValue[]
    name: string
    color: string
    selected: number
    titleWidth: number
    chartWidth: number
    onClick?: () => void
    isMouseMoveOpt?: boolean
    primaryElem: SVGTextElement
    secondaryElem: SVGTextElement
    className: string = 'empty'
    xScale: d3.ScaleLinear<number, number>
    yScale: d3.ScaleLinear<number, number>
    bisect: d3.Bisector<number, number>
    linePlot: LinePlot
    isBase: boolean
    smoothedValues: number[]

    constructor(opt: SparkLineOptions) {
        this.series = opt.series
        this.smoothedValues = opt.smoothedValues

        if (opt.selected == -1) {
            this.series = [this.series[this.series.length - 1]]
            this.smoothedValues = [this.smoothedValues[this.smoothedValues.length - 1]]
        }
        this.name = opt.name
        this.selected = opt.selected
        this.onClick = opt.onClick
        this.isMouseMoveOpt = opt.isMouseMoveOpt
        this.color = this.selected >= 0 ? opt.color : getBaseColor()
        this.chartWidth = Math.min(300, Math.round(opt.width * .60))
        this.titleWidth = (opt.width - this.chartWidth) / 2
        this.isBase = opt.isBase ?? false


        this.yScale = getScale(getExtent([this.series], d => d.value, true), -25)
        this.xScale = getScale(opt.stepExtent, this.chartWidth)

        this.bisect = d3.bisector(function (d: PointValue) {
            return d.step
        }).left

        if (this.onClick != null && this.selected >= 1) {
             this.className = 'selected'
        }
    }

    changeCursorValue(cursorStep?: number | null) {
        if (this.selected >= 0 && this.isMouseMoveOpt) {
            this.linePlot.renderIndicators(cursorStep)
            this.renderValue(cursorStep)
        }
    }

    renderValue(cursorStep?: number | null) {
        const index = this.selected >= 0 && this.isMouseMoveOpt ?
            getSelectedIdx(this.series, this.bisect, cursorStep) : this.series.length - 1
        const last = this.series[index]
        const lastSmoothed = this.smoothedValues[index]

        if (Math.abs(last.value - lastSmoothed) > Math.abs(last.value) / 1e6) {
            this.secondaryElem.textContent = formatFixed(last.value, 6)
        } else {
            this.secondaryElem.textContent = ''
        }
        this.primaryElem.textContent = formatFixed(lastSmoothed, 6)
    }

    render($: WeyaElementFunction) {
        $(`div.sparkline-list-item.list-group-item.${this.className}`, {on: {click: this.onClick}}, $ => {
            $(`div.sparkline-content`, {style: {width: `${Math.min(this.titleWidth * 2 + this.chartWidth, 450)}px`}}, $ => {
                if (this.onClick != null) {
                    if (this.selected == 1) {
                        $('span', '.fas.fa-eye.title.icon', '')
                    } else {
                        $('span', '.fas.fa-eye-slash.title.icon', '')
                    }
                }
                $('span', '.title', this.name, {style: {color: this.color}})
                $('svg.sparkline', {style: {width: `${this.chartWidth + this.titleWidth * 2}px`}, height: 36}, $ => {
                    $('g', {transform: `translate(${this.titleWidth}, 30)`}, $ => {
                        new LineFill({
                            series: this.series,
                            xScale: this.xScale,
                            yScale: this.yScale,
                            color: '#7f8c8d',
                            colorIdx: 9
                        }).render($)
                        this.linePlot = new LinePlot({
                            series: this.series,
                            xScale: this.xScale,
                            yScale: this.yScale,
                            color: '#7f8c8d',
                            isBase: this.isBase,
                            smoothedSeries: this.series
                        })
                        this.linePlot.render($)
                    })
                    $('g', {transform: `translate(${this.titleWidth * 2 + this.chartWidth}, ${0})`}, $ => {
                        this.secondaryElem = $('text', '.value-secondary', {
                            style: {fill: this.color},
                            transform: `translate(${0},${12})`
                        })
                        this.primaryElem = $('text', '.value-primary', {
                            style: {fill: this.color},
                            transform: `translate(${0},${29})`
                        })
                    })
                })
            })
        })

        this.renderValue()
    }
}
