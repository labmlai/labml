import {WeyaElementFunction} from '../../../../../lib/weya/weya'
import {ChartOptions} from '../types'
import {SeriesModel} from "../../../models/run"
import {defaultSeriesToPlot, getExtent} from "../utils"
import {SparkLine} from "./spark_line"
import ChartColors, {ChartColorsBase} from "../chart_colors"
import {DefaultLineGradient} from "../chart_gradients"
import DistributedChartColors from "../distributed_chart_colors";


interface SparkLinesOptions extends ChartOptions {
    plotIdx: number[]
    onSelect?: (i: number) => void
    isMouseMoveOpt?: boolean
    isDivergent?: boolean
    isDistributed?: boolean
}

export class SparkLines {
    series: SeriesModel[]
    plotIdx: number[]
    isEditable: boolean
    rowWidth: number
    minLastValue: number
    maxLastValue: number
    isMouseMoveOpt: boolean
    stepExtent: [number, number]
    colorIndices: number[] = []
    onSelect?: (i: number) => void
    sparkLines: SparkLine[] = []
    chartColors: ChartColorsBase
    isDivergent?: boolean
    isDistributed?: boolean
    uniqueItems: Map<string, number>

    constructor(opt: SparkLinesOptions) {
        this.series = opt.series
        this.plotIdx = opt.plotIdx
        this.onSelect = opt.onSelect
        this.isMouseMoveOpt = opt.isMouseMoveOpt
        this.isDistributed = opt.isDistributed

        const margin = Math.floor(opt.width / 64)
        this.rowWidth = Math.min(450, opt.width - 3 * margin)

        let lastValues: number[] = []
        let uniqueItemIdx = 0
        this.uniqueItems = new Map<string, number>()
        for (let s of this.series) {
            let series = s.series
            lastValues.push(series[series.length - 1].value)
            if (!this.uniqueItems.has(s.name)) {
                this.uniqueItems.set(s.name, uniqueItemIdx++)
            }
        }

        this.maxLastValue = Math.max(...lastValues)
        this.minLastValue = Math.min(...lastValues)

        this.stepExtent = getExtent(this.series.map(s => s.series), d => d.step)

        if (this.plotIdx.length === 0) {
            this.plotIdx = defaultSeriesToPlot(this.series)
        }

        for (let i = 0; i < this.plotIdx.length; i++) {
            if (this.plotIdx[i] >= 0) {
                this.colorIndices.push(i)
            } else {
                this.colorIndices.push(-1)
            }
        }

        if (this.isDistributed) {
            this.chartColors = new DistributedChartColors({nColors: this.uniqueItems.size, nShades: this.series.length})
        } else {
            this.chartColors = new ChartColors({nColors: this.series.length, isDivergent: opt.isDivergent})
        }
    }

    changeCursorValues = (cursorStep?: number | null) => {
        for (let sparkLine of this.sparkLines) {
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
                let sparkLine = new SparkLine({
                    name: s.name,
                    series: s.series,
                    selected: this.plotIdx[i],
                    stepExtent: this.stepExtent,
                    width: this.rowWidth,
                    onClick: onClick,
                    minLastValue: this.minLastValue,
                    maxLastValue: this.maxLastValue,
                    color: this.chartColors.getColor(this.colorIndices[i], this.uniqueItems.get(s.name) ?? 0),
                    isMouseMoveOpt: this.isMouseMoveOpt
                })
                this.sparkLines.push(sparkLine)
                sparkLine.render($)
            })
        })
    }
}