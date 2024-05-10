import {SeriesSmoothing} from "./smoothing_base"
import {PointValue} from "../../../models/run"

export class TwoSidedExponentialAverage extends SeriesSmoothing {
    protected smooth(): void {
        let smoothingFactor = 1 - this.smoothValue / 100

        for (let i = 0; i < this.indicators.length; i++) {
            let ind = this.indicators[i]

            if (ind.series.length < 2) {
                continue
            }

            let result: PointValue[] = []
            let forward_pass: number[] = []
            let lastSmoothed = ind.series[0].value
            for (let j = 0; j < ind.series.length; j++) {
                let smoothed = lastSmoothed * (1 - smoothingFactor) + ind.series[j].value * smoothingFactor
                forward_pass.push(smoothed)
                lastSmoothed = smoothed
            }

            let backward_pass: number[] = []
            lastSmoothed = ind.series[ind.series.length - 1].value
            for (let j = ind.series.length - 1; j >= 0; j--) {
                let smoothed = lastSmoothed * (1 - smoothingFactor) + ind.series[j].value * smoothingFactor
                backward_pass.push(smoothed)
                lastSmoothed = smoothed
            }
            backward_pass = backward_pass.reverse()

            for (let j = 0; j < ind.series.length; j++) {
                let smoothed = (forward_pass[j] + backward_pass[j]) / 2
                result.push({step: ind.series[j].step, value: ind.series[j].value, smoothed: smoothed,
                    lastStep: ind.series[j].lastStep})
            }

            ind.series = result
        }
    }
}