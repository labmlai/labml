import d3 from "../../d3"
import {ChartColorsBase} from "./chart_colors"

const DIVERGENT = d3.piecewise(d3.interpolateHcl, ["#ffa600", "#bc5090", "#003f5d"])
const DARK_DIVERGENT = d3.piecewise(d3.interpolateHcl, ["#ffa600", "#f73790", "#0088ce"])
const DIVERGENT_SECOND = d3.piecewise(d3.interpolateHcl, ["#965a01", "#6c0042", "#001d3b"])

interface ChartColorsOptions {
    nColors: number
    nShades: number
}

export default class DistributedChartColors implements ChartColorsBase {
    nColors: number
    nShades: number

    private readonly  lowerBounds: string[]
    private readonly upperBounds: string[]

    constructor(opt: ChartColorsOptions) {
        this.nColors = opt.nColors
        this.nShades = opt.nShades

        let colorScale: d3.ScaleLinear<number, string> =
            document.body.classList.contains('dark') ? DIVERGENT : DARK_DIVERGENT
        let secondColorScale: d3.ScaleLinear<number, string> = DIVERGENT_SECOND

        this.lowerBounds = []
        this.upperBounds = []

        for (let n=0; n<this.nColors; ++n) {
            this.lowerBounds.push(colorScale(n / (Math.max(1, this.nColors - 1))))
            this.upperBounds.push(secondColorScale(n / (Math.max(1, this.nColors - 1))))
        }
    }

    getColor(nShade: number, nColor: number): string {
        let colorScale = d3.piecewise(d3.interpolateHcl, [this.lowerBounds[nColor], this.upperBounds[nColor]])
        return colorScale(nShade / (Math.max(1, this.nShades - 1)))
    }

    getColors(shade: number): string[] {
        return []
    }
}
