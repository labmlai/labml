import d3 from "../../../d3"
import {WeyaElementFunction} from '../../../../../lib/weya/weya'
import {PointValue, SeriesModel} from "../../../models/run"
import {getBaseColor} from "../constants"
import {getExtent, getScale, getSelectedIdx} from "../utils"
import {LineFill, LinePlot} from "../lines/plot"
import {numberWithCommas, scientificFormat} from "../../../utils/value"
import Timeout = NodeJS.Timeout

export interface SparkLineOptions {
    name: string
    dynamic_type: string
    range: [number, number]
    series: PointValue[]
    sub: SeriesModel
    width: number
    stepExtent: [number, number]
    selected: number
    minLastValue: number
    maxLastValue: number
    onClick?: () => void
    onEdit: () => void
    isMouseMoveOpt?: boolean
    color: string
}

export class EditableSparkLine {
    name: string
    dynamic_type: string
    range: [number, number]
    series: PointValue[]
    sub: SeriesModel
    minLastValue: number
    maxLastValue: number
    color: string
    selected: number
    titleWidth: number
    chartWidth: number
    onClick?: () => void
    onEdit: () => void
    isMouseMoveOpt?: boolean
    className: string = 'empty'
    xScale: d3.ScaleLinear<number, number>
    yScale: d3.ScaleLinear<number, number>
    bisect: d3.Bisector<number, number>
    linePlot: LinePlot
    inputRangeElem: HTMLInputElement
    inputValueElem: HTMLInputElement
    inputElements: HTMLDivElement
    primaryElem: SVGTextElement
    lastChanged: number
    inputTimeout: Timeout

    constructor(opt: SparkLineOptions) {
        this.name = opt.name
        this.dynamic_type = opt.dynamic_type
        this.range = opt.range
        this.series = opt.series
        this.sub = opt.sub
        this.selected = opt.selected
        this.onClick = opt.onClick
        this.onEdit = opt.onEdit
        this.isMouseMoveOpt = opt.isMouseMoveOpt
        this.color = this.selected >= 0 ? opt.color : getBaseColor()
        this.chartWidth = Math.min(300, Math.round(opt.width * .60))
        this.titleWidth = (opt.width - this.chartWidth) / 2
        this.minLastValue = opt.minLastValue
        this.maxLastValue = opt.maxLastValue
        this.inputTimeout = null

        this.yScale = getScale(getExtent([this.series], d => d.value, true), -25)
        this.xScale = getScale(opt.stepExtent, this.chartWidth)

        this.bisect = d3.bisector(function (d: PointValue) {
            return d.step
        }).left

        if (this.onClick != null && this.selected >= 0) {
            this.className = 'selected'
        }

        if (this.onClick != null) {
            this.className += '.list-group-item-action'
        }
    }

    isIntegerType() {
        return this.dynamic_type === 'int'
    }

    changeCursorValue(cursorStep?: number | null) {
        if (this.selected >= 0) {
            this.linePlot.renderCursorCircle(cursorStep)
            this.renderTextValue(cursorStep)
        }
    }

    formatNumber(value: number) {
        if (value >= 10000 || value < 0.001) {
            return scientificFormat(value)
        }

        let decimals
        if (this.dynamic_type === 'float') {
            decimals = 3
        } else {
            decimals = 0
        }

        let str = value.toFixed(decimals)

        return numberWithCommas(str)
    }

    renderTextValue(cursorStep?: number | null) {
        const last = this.series[this.selected >= 0 || this.isMouseMoveOpt ?
            getSelectedIdx(this.series, this.bisect, cursorStep) : this.series.length - 1]

        this.primaryElem.textContent = this.formatNumber(last.value)
    }

    renderInputValues() {
        let s = this.sub ? this.sub.series : this.series
        const last = s[s.length - 1]
        this.inputValueElem.value = this.formatNumber(last.value)
        this.updateSliderConfig(last.value)

        this.lastChanged = undefined
    }

    render($: WeyaElementFunction) {
        $(`div.sparkline-list-item.list-group-item.${this.className}`, {on: {click: this.onClick}}, $ => {
            $('div.sparkline-content', {style: {width: `${Math.min(this.titleWidth * 2 + this.chartWidth, 450)}px`}}, $ => {
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
                            color: '#7f8c8d'
                        })
                        this.linePlot.render($)
                    })
                    $('g', {transform: `translate(${this.titleWidth * 2 + this.chartWidth}, ${0})`}, $ => {
                        this.primaryElem = $('text', '.value-primary', {
                            style: {fill: this.color},
                            transform: `translate(${0},${20})`
                        })
                    })
                })
                this.inputElements = $('div', '.mt-1', {style: {width: `${Math.min(this.titleWidth * 2 + this.chartWidth, 450)}px`}}, $ => {
                    $('span', ' ', {style: {width: `${this.titleWidth}px`}})
                    this.inputRangeElem = $('input', '.slider', {
                        type: "range",
                        style: {width: `${this.chartWidth}px`},
                    })
                    $('span.input-container', {style: {width: `${this.titleWidth}px`}}, $ => {
                        $('span.input-content.float-right', $ => {
                            this.inputValueElem = $('input', '.text-end', {
                                style: {
                                    height: '36px',
                                    width: `${this.titleWidth/1.1}px`,
                                    padding: '0px'
                                }
                            })
                        })
                    })
                })
            })
        })

        this.inputRangeElem.addEventListener('click', this.onInputElemClick.bind(this))
        this.inputValueElem.addEventListener('click', this.onInputElemClick.bind(this))
        this.inputRangeElem.addEventListener('input', this.onSliderChange.bind(this))
        this.inputValueElem.addEventListener('input', this.onInputChange.bind(this))

        this.renderInputValues()
        this.renderTextValue()

        if (this.className.includes('selected')) {
            this.inputElements.style.display = 'block'
        } else {
            this.inputElements.style.display = 'none'
        }
    }

    onInputElemClick(e: Event) {
        e.preventDefault()
        e.stopPropagation()
    }

    onSliderChange() {
        let strNumber = this.inputRangeElem.value
        this.lastChanged = Number(strNumber)
        this.inputValueElem.value = this.formatNumber(this.lastChanged)
        this.onEdit()
    }

    onInputChange() {
        this.lastChanged = Number(this.inputValueElem.value)
        this.onEdit()
    }

    updateSliderConfig(value) {
        let min: number = value / 5
        let max: number = value * (9 / 5)

        if (this.range) {
            min = this.range[0]
            max = this.range[1]
            value = (min + max) / 2
        }

        let step: number = (min + max) / 20

        if (this.isIntegerType()) {
            min = Math.round(min)
            max = Math.round(max)
            value = Math.round(value)
            step = Math.max(1, Math.round(step))
        }

        this.inputRangeElem.setAttribute("min", `${min}`)
        this.inputRangeElem.setAttribute("max", `${max}`)
        this.inputRangeElem.setAttribute("step", `${step}`)
        this.inputRangeElem.setAttribute("value", `${value}`)
    }

    getInput() {
        return this.lastChanged
    }

    getInputValidation() {
        let res = ''

        if (isNaN(this.lastChanged)) {
            res = 'not a number'
        } else if (this.isIntegerType() && !Number.isInteger(this.lastChanged)) {
            res = 'should be an integer'
        }

        return res
    }
}
