import d3 from "../../d3"
import {WeyaElementFunction} from "../../../../lib/weya/weya"

interface AxisOptions {
    chartId: string
    scale: d3.ScaleLinear<number, number>
    specifier?: string
    numTicks?: number
}

export class RightAxis {
    scale: d3.ScaleLinear<number, number>
    specifier?: string
    numTicks?: number
    id: string
    axis

    constructor(opt: AxisOptions) {
        this.specifier = opt.specifier !== undefined ? opt.specifier : ""
        this.numTicks = opt.numTicks !== undefined ? opt.numTicks : 5
        this.id = `${opt.chartId}_axis_right`
        this.axis = d3.axisRight(opt.scale as d3.AxisScale<d3.AxisDomain>).ticks(this.numTicks, this.specifier)
    }

    render($: WeyaElementFunction) {
        $('g', {id: this.id})

        let layer = d3.select(`#${this.id}`)
        layer.selectAll('g').remove()
        layer.append('g').call(this.axis)
    }
}

export class BottomAxis {
    scale: d3.ScaleLinear<number, number>
    specifier?: string
    id: string
    axis

    constructor(opt: AxisOptions) {
        this.specifier = opt.specifier !== undefined ? opt.specifier : ".2s"
        this.id = `${opt.chartId}_axis_bottom`
        this.axis = d3.axisBottom(opt.scale as d3.AxisScale<d3.AxisDomain>).ticks(5, this.specifier)
    }

    render($: WeyaElementFunction) {
        $('g', {id: this.id})

        let layer = d3.select(`#${this.id}`)
        layer.selectAll('g').remove()
        layer.append('g').call(this.axis)
    }
}

interface TimeAxisOptions {
    chartId: string
    scale: d3.ScaleTime<number, number>
    numTicks?: number
}

export class BottomTimeAxis {
    scale: d3.ScaleTime<number, number>
    numTicks?: number
    id: string
    axis

    constructor(opt: TimeAxisOptions) {
        this.numTicks = opt.numTicks | 5
        this.id = `${opt.chartId}_axis_bottom`
        this.axis = d3.axisBottom(opt.scale as d3.AxisScale<d3.AxisDomain>).ticks(this.numTicks, d3.timeFormat("%b-%d:%H:%M"))
    }

    render($: WeyaElementFunction) {
        $('g', {id: this.id})

        let layer = d3.select(`#${this.id}`)
        layer.selectAll('g').remove()
        layer.append('g').call(this.axis)
    }
}
