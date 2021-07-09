import {WeyaElementFunction} from "../../../../lib/weya/weya"
import ChartColors from "./chart_colors"

interface LabelsOptions {
    labels: string[]
    isDivergent?: boolean
    chartColors?: ChartColors
}

export class Labels {
    labels: string[]
    chartColors: ChartColors

    constructor(opt: LabelsOptions) {
        this.labels = opt.labels
        this.chartColors = opt.chartColors ? opt.chartColors : new ChartColors({
            nColors: this.labels.length,
            isDivergent: opt.isDivergent
        })
    }

    render($: WeyaElementFunction) {
        $('div.text-center.labels.text-secondary',
            $ => {
                this.labels.map((label, i) => {
                    $('span', $ => {
                        $('div.box', {style: {background: this.chartColors.getColor(i)}})
                    })
                    $('span', label)
                })
            })
    }
}