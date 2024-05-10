import {SeriesSmoothing} from "./smoothing_base"
import {PointValue} from "../../../models/run"

export class TwoSidedExponentialAverage extends SeriesSmoothing {
    protected smooth(): void {
        let smoothingFactor = this.smoothValue / 100

        for (let i = 0; i < this.indicators.length; i++) {
            let ind = this.indicators[i]

            if (ind.series.length < 2) {
                continue
            }

            let result: PointValue[] = []

            let f_sum: number[] = []
            let b_sum: number[] = []
            let f_weights: number[] = []
            let b_weights: number[] = []

            let last_sum = 0
            let last_weight = 0
            for (let j = 0; j < ind.series.length; j++) {
                f_sum.push(last_sum * smoothingFactor + ind.series[j].value)
                f_weights.push(last_weight * smoothingFactor + 1)

                last_sum = f_sum[j]
                last_weight = f_weights[j]
            }

            last_sum = 0
            last_weight = 0
            for (let j = ind.series.length - 1; j >= 0; j--) {
                b_sum.push(last_sum * smoothingFactor + ind.series[j].value)
                b_weights.push(last_weight * smoothingFactor + 1)

                last_sum = b_sum[b_sum.length - 1]
                last_weight = b_weights[b_weights.length - 1]
            }
            b_weights.reverse()
            b_sum.reverse()

            for (let j = 0; j < ind.series.length-1; j++) {
                let smoothed = (f_sum[j] + smoothingFactor * b_sum[j+1]) /
                    (f_weights[j] + smoothingFactor * b_weights[j+1])
                result.push({step: ind.series[j].step, value: ind.series[j].value,
                    smoothed: smoothed, lastStep: ind.series[j].lastStep})
            }
            result.push({step: ind.series[ind.series.length-1].step, value: ind.series[ind.series.length-1].value,
                smoothed: f_sum[ind.series.length-1] / f_weights[ind.series.length-1],
                lastStep: ind.series[ind.series.length-1].lastStep})



            ind.series = result
        }
    }
}