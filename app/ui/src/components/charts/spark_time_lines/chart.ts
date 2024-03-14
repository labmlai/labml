import {WeyaElementFunction} from '../../../../../lib/weya/weya'
import {ChartOptions} from '../types'
import {getExtent, toDate} from "../utils"
import {SparkTimeLine} from "./spark_time_line"
import ChartColors from "../chart_colors"
import {DefaultLineGradient} from "../chart_gradients"
import {getBaseColor} from '../constants'
import {Indicator} from "../../../models/run"


interface SparkTimeLinesOptions extends ChartOptions {
    plotIdx: number[]
    onSelect?: (i: number) => void
    isDivergent?: boolean
    isColorless?: boolean
}

export class SparkTimeLines {
    series: Indicator[]
    plotIdx: number[]
    rowWidth: number
    minLastValue: number
    maxLastValue: number
    stepExtent: [number, number]
    colorIndices: number[] = []
    onSelect?: (i: number) => void
    sparkTimeLines: SparkTimeLine[] = []
    chartColors: ChartColors
    isDivergent?: boolean
    isColorless: boolean

    constructor(opt: SparkTimeLinesOptions) {
        this.series = opt.series
        this.plotIdx = opt.plotIdx
        this.onSelect = opt.onSelect
        this.isColorless = opt.isColorless || false

        const margin = Math.floor(opt.width / 64)
        this.rowWidth = Math.min(450, opt.width - 3 * margin)

        let lastValues: number[] = []
        for (let s of this.series) {
            let series = s.series
            lastValues.push(series[series.length - 1].value)
        }

        this.maxLastValue = Math.max(...lastValues)
        this.minLastValue = Math.min(...lastValues)

        this.stepExtent = getExtent(this.series.map(s => s.series), d => d.step)

        for (let i = 0; i < this.plotIdx.length; i++) {
            if (this.plotIdx[i] >= 0) {
                this.colorIndices.push(i)
            } else {
                this.colorIndices.push(-1)
            }
        }

        this.chartColors = new ChartColors({nColors: this.series.length, isDivergent: opt.isDivergent})
    }

    changeCursorValues = (cursorStep?: Date | null) => {
        for (let sparkLine of this.sparkTimeLines) {
            sparkLine.changeCursorValue(cursorStep)
        }
    }

    render($: WeyaElementFunction) {
        $('div.sparkline-list.list-group', $ => {
            this.series.map((s, i) => {
                $('svg', {style: {height: `${1}px`}}, $ => {
                    new DefaultLineGradient().render($)
                })
                let onClick
                if (this.onSelect != null) {
                    onClick = this.onSelect.bind(null, i)
                }
                let sparkTimeLine = new SparkTimeLine({
                    name: s.name,
                    series: s.series,
                    selected: this.plotIdx[i],
                    stepExtent: [toDate(this.stepExtent[0]), toDate(this.stepExtent[1])],
                    width: this.rowWidth,
                    onClick: onClick,
                    minLastValue: this.minLastValue,
                    maxLastValue: this.maxLastValue,
                    color: this.isColorless ? getBaseColor(): this.chartColors.getColor(this.colorIndices[i]),
                })
                this.sparkTimeLines.push(sparkTimeLine)
                sparkTimeLine.render($)
            })
        })
    }
}
