import d3 from "../../d3"

import {PointValue, SeriesModel} from "../../models/run"
import {OUTLIER_MARGIN} from "./constants"

export function getExtentWithoutOutliers(series: PointValue[], func: (d: PointValue) => number): [number, number] {
    let values = series.map(func)
    values.sort((a, b) => a - b)
    if (values.length === 0) {
        return [0, 0]
    }
    if (values.length < 10) {
        return [values[0], values[values.length - 1]]
    }
    let extent = [0, values.length - 1]
    let margin = Math.floor(values.length * OUTLIER_MARGIN)
    let stdDev = d3.deviation(values.slice(margin, values.length - margin))
    if (stdDev == null) {
        stdDev = (values[values.length - margin - 1] - values[margin]) / 2
    }
    for (; extent[0] < margin; extent[0]++) {
        if (values[extent[0]] + stdDev * 2 > values[margin]) {
            break
        }
    }
    for (; extent[1] > values.length - margin - 1; extent[1]--) {
        if (values[extent[1]] - stdDev * 2 < values[values.length - margin - 1]) {
            break
        }
    }

    return [values[extent[0]], values[extent[1]]]
}

export function getExtent(series: PointValue[][], func: (d: PointValue) => number, forceZero: boolean = false, skipZero: boolean = false): [number, number] {
    let extent = getExtentWithoutOutliers(series[0], func)

    for (let s of series) {
        let e = getExtentWithoutOutliers(s, func)

        if (skipZero && e[0] === 0) {
            continue
        }

        extent[0] = Math.min(e[0], extent[0])
        extent[1] = Math.max(e[1], extent[1])
    }

    if (skipZero) {
        return extent
    }

    if (forceZero || (extent[0] > 0 && extent[0] / extent[1] < 0.1)) {
        extent[0] = Math.min(0, extent[0])
    }

    return extent
}

export function getScale(extent: [number, number], size: number, isNice: boolean = true): d3.ScaleLinear<number, number> {
    if (isNice) {
        return d3.scaleLinear()
            .domain(extent).nice()
            .range([0, size])
    } else {
        return d3.scaleLinear()
            .domain(extent)
            .range([0, size])
    }
}

export function getLogScale(extent: [number, number], size: number): d3.ScaleLogarithmic<number, number> {
    return d3.scaleLog()
        .domain(extent)
        .range([0, size])
}

export function getTimeScale(extent: [Date, Date], size: number): d3.ScaleTime<number, number> {
    return d3.scaleTime()
        .domain(extent)
        .range([0, size])
}

export function toDate(time: number) {
    return new Date(time * 1000)
}

export function defaultSeriesToPlot(series: SeriesModel[]) {
    let count = 0
    let plotIdx = []
    for (let s of series) {
        let name = s.name.split('.')
        if (name[0] === 'loss') {
            plotIdx.push(count)
            count++
        } else {
            plotIdx.push(-1)
        }
    }

    return plotIdx
}

export function toPointValue(s: SeriesModel) {
    let res: PointValue[] = []
    for (let i = 0; i < s.step.length; ++i) {
        res.push({step: s.step[i], value: s.value[i], smoothed: s.smoothed[i]})
    }

    return res
}

export function toPointValues(track: SeriesModel[]) {
    let series: SeriesModel[] = [...track]
    for (let s of series) {
        s.series = toPointValue(s)
    }

    return series
}

export function getSelectedIdx(series: any[], bisect: typeof d3.bisect, currentX?: any | null, stepKey: string = 'step') {
    let idx = series.length - 1
    if (currentX != null) {
        idx = bisect(series, currentX)
        if (idx < series.length) {
            if (idx !== 0) {
                idx = Math.abs(currentX - series[idx - 1][stepKey]) > Math.abs(currentX - series[idx][stepKey]) ?
                    idx : idx - 1
            }

            return idx
        } else {
            return series.length - 1
        }
    }

    return idx
}

export function getChartType(index: number): 'log' | 'linear' {
    return index === 0 ? 'linear' : 'log'
}

export function trimSteps(series: SeriesModel[], min: number, max: number) : SeriesModel[] {
    // let minStep = Math.min(...thresholdSeries.map(s => s.series[0].step))
    // let maxStep = Math.max(...thresholdSeries.map(s => s.series[s.series.length - 1].step))

    return series.map(s => {
        let res = {...s}
        res.series = s.series.filter(p => {
            return (p.step >= min || min == -1) && (p.step <= max || max == -1)
        })

        return res
    })
}