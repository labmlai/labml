import {SeriesSmoothing, SeriesSmoothingOptions} from "./smoothing_base"
import {Indicator, PointValue} from "../../../models/run"
import {mapRange} from "../utils";

export class MovingAverage extends SeriesSmoothing {
    private readonly trimSmoothEnds: boolean
    private readonly smoothWindow: number[]

    constructor(opt: SeriesSmoothingOptions, trimSmoothEnds: boolean ) {
        super(opt)
        this.trimSmoothEnds = trimSmoothEnds
        this.smoothWindow = this.getSmoothWindow(opt.indicators, opt.smoothValue)
    }

    protected smooth(): void {
        for (let i = 0; i < this.indicators.length; i++) {
            let ind = this.indicators[i]
            let windowSize = this.smoothWindow[i]

            let result: PointValue[] = []
            windowSize = ~~windowSize
            let extraWindow = windowSize / 2
            extraWindow = ~~extraWindow

            let count = 0
            let total = 0

            for (let i = 0; i < ind.series.length + extraWindow; i++) {
                let j = i - extraWindow
                if (i < ind.series.length) {
                    total += ind.series[i].value
                    count++
                }
                if (j - extraWindow - 1 >= 0) {
                    total -= ind.series[j - extraWindow - 1].value
                    count--
                }
                if (j>=0) {
                    result.push({step: ind.series[j].step, value: ind.series[j].value, smoothed: total / count,
                        lastStep: ind.series[j].lastStep})
                }
            }
            ind.series = result
        }
    }

    private getSmoothWindow(indicators: Indicator[], smoothValue: number): number[] {
        let maxRange: number = Number.MIN_SAFE_INTEGER
        for (let ind of indicators) {
            if (ind.series.length > 1 && !ind.is_summary) {
                maxRange = Math.max(maxRange, ind.series[ind.series.length - 1].step - ind.series[0].step)
            }
        }
        if (maxRange == Number.MIN_SAFE_INTEGER) { // all single points. -> can't smooth
            let stepRange = []
            for (let _ of indicators) {
                stepRange.push(1)
            }
            return stepRange
        }

        let smoothRange = mapRange(smoothValue, 1, 100, 1, 2*maxRange)

        let stepRange = []

        for (let ind of indicators) {
            if (ind.series.length >= 2 && !ind.is_summary) {
                let stepGap = ind.series[1].step - ind.series[0].step
                let numSteps = Math.max(1, Math.ceil(smoothRange / stepGap))
                stepRange.push(numSteps)
            } else { // can't smooth - just a single point
                stepRange.push(1)
            }
        }

        return stepRange
    }

    override trim(): void {
        this.indicators.forEach((ind, i) => {
            let localSmoothWindow = Math.floor(this.smoothWindow[i] / 2) // remove half from each end

            if (ind.series.length <= 1) {
                localSmoothWindow = 0
            } else if (this.smoothWindow[i] >= ind.series.length) {
                localSmoothWindow = Math.floor(ind.series.length/2)
            }

            let localMin = this.min
            let localMax = this.max

            if (localMin == -1) {
                localMin = ind.series[0].step
            }
            if (this.trimSmoothEnds) {
                localMin = Math.max(localMin, ind.series[localSmoothWindow].step)
            }

            if (localMax == -1) {
                localMax = ind.series[ind.series.length - 1].step
            }
            if (this.trimSmoothEnds) {
                localMax = Math.min(localMax, ind.series[ind.series.length - 1 - localSmoothWindow +
                (ind.series.length%2 == 0 && localSmoothWindow != 0 ? 1 : 0)].step) // get the mid value for even length series
            }

            localMin = Math.floor(localMin) - 0.5
            localMax = Math.ceil(localMax) + 0.5

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