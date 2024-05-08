import {Indicator} from "../../../models/run";

export interface SeriesSmoothingOptions {
    indicators: Indicator[]
    smoothValue: number
    min: number
    max: number
    currentIndicatorLength: number
}

export abstract class SeriesSmoothing {
    protected indicators: Indicator[]
    protected readonly smoothValue: number
    protected min: number
    protected max: number
    protected currentIndicatorLength: number

    protected constructor(opt: SeriesSmoothingOptions) {
        this.indicators = opt.indicators
        this.smoothValue = opt.smoothValue
        this.min = opt.min
        this.max = opt.max
        this.currentIndicatorLength = opt.currentIndicatorLength
    }

    public smoothAndTrim(): Indicator[][] {
        this.smooth()
        this.trim()

        return [this.indicators.slice(0, this.currentIndicatorLength),
            this.indicators.slice(this.currentIndicatorLength)]
    }

    protected abstract smooth(): void

    protected trim(): void {
        this.indicators.forEach((ind, i) => {
            if (ind.series.length == 0) {
                return
            }

            let localMin = this.min == -1 ? ind.series[0].step : this.min
            let localMax = this.max == -1 ? ind.series[ind.series.length - 1].step : this.max

            let minIndex = ind.series.length - 1
            let maxIndex = 0

            for (let i = 0; i < ind.series.length; i++) {
                let p = ind.series[i]
                if (p.step >= localMin && p.step <= localMax) {
                    minIndex = Math.min(i, minIndex)
                    maxIndex = Math.max(i, maxIndex)
                }
            }

            ind.lowTrimIndex = minIndex
            ind.highTrimIndex = maxIndex
        })
    }
}