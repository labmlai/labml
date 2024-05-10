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
            let last_step = ind.series[0].step
            for (let j = 0; j < ind.series.length; j++) {
                let smooth_gap = Math.pow(smoothingFactor, Math.abs(ind.series[j].step - last_step))
                f_sum.push(last_sum * smooth_gap + ind.series[j].value)
                f_weights.push(last_weight * smooth_gap + 1)

                last_sum = f_sum[j]
                last_weight = f_weights[j]
                last_step = ind.series[j].step
            }

            last_sum = 0
            last_weight = 0
            last_step = ind.series[ind.series.length - 1].step
            for (let j = ind.series.length - 1; j >= 0; j--) {
                let smooth_gap = Math.pow(smoothingFactor, Math.abs(ind.series[j].step - last_step))
                b_sum.push(last_sum * smooth_gap + ind.series[j].value)
                b_weights.push(last_weight * smooth_gap + 1)

                last_sum = b_sum[b_sum.length - 1]
                last_weight = b_weights[b_weights.length - 1]
                last_step = ind.series[j].step
            }
            b_weights.reverse()
            b_sum.reverse()

            for (let j = 0; j < ind.series.length; j++) {
                let smoothed = (f_sum[j] + b_sum[j] - ind.series[j].value) /
                    (f_weights[j] + b_weights[j] - 1)
                result.push({step: ind.series[j].step, value: ind.series[j].value,
                    smoothed: smoothed, lastStep: ind.series[j].lastStep})
            }

            ind.series = result
        }
    }
}