import d3 from "../../d3"

const LIGHT_SINGLE_HUE = d3.piecewise(d3.interpolateHsl, ["#004c6d", "#c1e7ff"])
const DARK_SINGLE_HUE = d3.piecewise(d3.interpolateHsl, ["#c1e7ff", "#004c6d"])

interface ChartColorsOptions {
    nColors: number
    secondNColors?: number
    isDivergent?: boolean
}

const DIVERGENT = d3.piecewise(d3.interpolateHcl, ["#ffa600", "#bc5090", "#003f5d"])
const DARK_DIVERGENT = d3.piecewise(d3.interpolateHcl, ["#FFCF75", "#DBA1C4", "#1DB7FF"])
const DIVERGENT_SECOND = d3.piecewise(d3.interpolateHcl, ["#BFA08D", "#B790B8", "#216AB9"])
const DARK_DIVERGENT_SECOND = d3.piecewise(d3.interpolateHcl, ["#FF7606", "#D03C7A", "#3C608C"])

export interface ChartColorsBase {
    getColors(): string[]

    getColor(n: number): string
}

export default class ChartColors implements ChartColorsBase {
    private readonly colors: string[] = []
    private readonly secondColors: string[] = []

    constructor(opt: ChartColorsOptions) {
        let nColors = opt.nColors
        let secondNColors = opt.secondNColors ?? 0

        let colorScale: d3.ScaleLinear<number, string>
        let secondColorScale: d3.ScaleLinear<number, string>

        if (opt.isDivergent) {
            if (document.body.classList.contains('dark')) {
                colorScale = DARK_DIVERGENT
                secondColorScale = DARK_DIVERGENT_SECOND
            } else {
                colorScale = DIVERGENT
                secondColorScale = DIVERGENT_SECOND
            }
        } else {
            if (document.body.classList.contains('light')) {
                colorScale = LIGHT_SINGLE_HUE
                secondColorScale = LIGHT_SINGLE_HUE
            } else {
                colorScale = DARK_SINGLE_HUE
                secondColorScale = DARK_SINGLE_HUE
            }
        }

        for (let n = 0; n < nColors; ++n) {
            this.colors.push(colorScale(n / (Math.max(1, nColors - 1))))
        }
        for (let n = 0; n < secondNColors; ++n) {
            this.secondColors.push(secondColorScale(n / (Math.max(1, secondNColors - 1))))
        }
    }

    getColor(i: number): string {
        return this.colors[i]
    }

    getSecondColor(i: number) {
        return this.secondColors[i]
    }

    getColors() {
        return this.colors
    }

    getSecondColors() {
        return this.secondColors
    }
}
