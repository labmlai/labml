import d3 from "../../d3"
import {SeriesModel} from "../../models/run"

export interface ChartOptions {
    series: SeriesModel[]
    width: number
}

export interface PlotOptions {
    yScale: d3.ScaleLinear<number, number>
    color: string
}

export interface FillOptions extends PlotOptions {
    colorIdx: number
}
