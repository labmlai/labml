import d3 from "../../d3"
import {Indicator} from "../../models/run"

export interface ChartOptions {
    series: Indicator[]
    width: number
}

export interface PlotOptions {
    yScale: d3.ScaleLinear<number, number>
    color: string
}

export interface FillOptions extends PlotOptions {
    colorIdx: number
}
