import {SeriesSmoothing} from "./smoothing_base";
import {PointValue} from "../../../models/run";

export class ExponentialMovingAverage extends SeriesSmoothing {
    protected smooth(): void {
        let smoothingFactor = 1 - this.smoothValue / 100

        for (let i = 0; i < this.indicators.length; i++) {
            let ind = this.indicators[i]

            if (ind.series.length == 0) {
                continue
            }

            let result: PointValue[] = []

            let lastSmoothed = ind.series[0].value
            for (let j = 0; j < ind.series.length; j++) {
                let smoothed = lastSmoothed * (1 - smoothingFactor) + ind.series[j].value * smoothingFactor
                result.push({step: ind.series[j].step, value: ind.series[j].value, smoothed: smoothed,
                    lastStep: ind.series[j].lastStep})
                lastSmoothed = smoothed
            }

            ind.series = result
        }
    }
}