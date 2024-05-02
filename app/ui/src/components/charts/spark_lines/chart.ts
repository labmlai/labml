import {WeyaElementFunction} from '../../../../../lib/weya/weya'
import {ChartOptions} from '../types'
import {getExtent} from "../utils"
import {SparkLine} from "./spark_line"
import ChartColors from "../chart_colors"
import {DefaultLineGradient} from "../chart_gradients"
import {Indicator} from "../../../models/run";

interface CompareSparkLinesOptions extends ChartOptions {
    baseSeries?: Indicator[]
    plotIdx: number[]
    basePlotIdx?: number[]
    onSelect?: (i: number) => void
    onBaseSelect?: (i: number) => void
    isMouseMoveOpt?: boolean
    isDivergent?: boolean
    onlySelected?: boolean
    smoothValue: number
}

export class SparkLines {
    currentSeries: Indicator[]
    baseSeries: Indicator[]
    currentPlotIdx: number[]
    basePlotIdx: number[]
    isEditable: boolean
    rowWidth: number
    isMouseMoveOpt: boolean
    stepExtent: [number, number]
    onCurrentSelect?: (i: number) => void
    onBaseSelect?: (i: number) => void
    sparkLines: SparkLine[] = []
    chartColors: ChartColors
    isDivergent?: boolean
    uniqueItems: Map<string, number>
    onlySelected: boolean

    constructor(opt: CompareSparkLinesOptions) {
        this.currentSeries = opt.series
        this.baseSeries = opt.baseSeries ?? []
        this.currentPlotIdx = opt.plotIdx
        this.basePlotIdx = opt.basePlotIdx ?? []
        this.onCurrentSelect = opt.onSelect
        this.onBaseSelect = opt.onBaseSelect
        this.isMouseMoveOpt = opt.isMouseMoveOpt
        this.uniqueItems = new Map<string, number>()
        this.onlySelected = opt.onlySelected ?? false

        let idx = 0
        for (let s of this.currentSeries.concat(this.baseSeries)) {
            if (!this.uniqueItems.has(s.name)) {
                this.uniqueItems.set(s.name, idx++)
            }
        }

        const margin = Math.floor(opt.width / 64)
        this.rowWidth = Math.min(450, opt.width - Math.max(3 * margin, 60))

        this.stepExtent = getExtent(this.currentSeries.concat(this.baseSeries).map(s => s.series), d => d.step)

        this.chartColors = new ChartColors({nColors: this.uniqueItems.size, secondNColors: this.uniqueItems.size, isDivergent: opt.isDivergent})
    }

    changeCursorValues = (cursorStep?: number | null) => {
        for (let sparkLine of this.sparkLines) {
            sparkLine.changeCursorValue(cursorStep)
        }
    }

    render($: WeyaElementFunction) {
        this.sparkLines = []
        $('div.sparkline-list.list-group', $ => {
            this.currentSeries.map((s, i) => {
                if (this.onlySelected && this.currentPlotIdx[i]!=1)
                    return
                $('svg', {style: {height: `${1}px`}}, $ => {
                    new DefaultLineGradient().render($)
                })
                let onClick
                if (this.onCurrentSelect != null) {
                    onClick = this.onCurrentSelect.bind(null, i)
                }
                let sparkLine = new SparkLine({
                    name: s.name,
                    series: s.series,
                    selected: this.currentPlotIdx[i],
                    stepExtent: this.stepExtent,
                    width: this.rowWidth,
                    onClick: onClick,
                    color: this.chartColors.getColor(this.uniqueItems.get(s.name)),
                    isMouseMoveOpt: this.isMouseMoveOpt,
                    isComparison: this.baseSeries.length > 0
                })
                this.sparkLines.push(sparkLine)
            })
            this.baseSeries.map((s, i) => {
                if (this.onlySelected && this.basePlotIdx[i]!=1)
                    return
                $('svg', {style: {height: `${1}px`}}, $ => {
                    new DefaultLineGradient().render($)
                })
                let onClick
                if (this.onBaseSelect != null) {
                    onClick = this.onBaseSelect.bind(null, i)
                }
                let sparkLine = new SparkLine({
                    name: s.name,
                    series: s.series,
                    selected: this.basePlotIdx[i],
                    stepExtent: this.stepExtent,
                    width: this.rowWidth,
                    onClick: onClick,
                    color: this.chartColors.getSecondColor(this.uniqueItems.get(s.name)),
                    isMouseMoveOpt: this.isMouseMoveOpt,
                    isBase: true,
                    isComparison: this.baseSeries.length > 0,
                })
                this.sparkLines.push(sparkLine)
            })
            this.sparkLines.sort((a, b) => a.name.localeCompare(b.name))
            this.sparkLines.map(sparkLine => {
                sparkLine.render($)
            })
        })
    }
}
