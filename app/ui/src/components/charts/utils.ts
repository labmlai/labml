import d3 from "../../d3"

import {Indicator, IndicatorModel, PointValue} from "../../models/run"
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
    if (series.length === 0) {
        return [0, 0]
    }

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

export function mapRange(value: number, fromSource: number, toSource: number, fromTarget: number, toTarget: number) {
    return (value - fromSource) / (toSource - fromSource) * (toTarget - fromTarget) + fromTarget;
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

export function smoothSeries(series: PointValue[], windowSize: number): PointValue[] {
    let result: PointValue[] = []
    windowSize = ~~windowSize
    let extraWindow = windowSize / 2
    extraWindow = ~~extraWindow
    if (series.length <= windowSize) {
        return series
    }

    let count = 0
    let total = 0

    for (let i = 0; i < series.length + extraWindow; i++) {
        let j = i - extraWindow
        if (i < series.length) {
            total += series[i].value
            count++
        }
        if (j - extraWindow - 1 >= 0) {
            total -= series[j - extraWindow - 1].value
            count--
        }
        if (j>=0) {
            result.push({step: series[j].step, value: series[j].value, smoothed: total / count, lastStep: series[j].lastStep})
        }
    }
    return result
}

export function fillPlotPreferences(series: Indicator[], currentPlotIdx: number[] = []) {
    if (currentPlotIdx.length != 0) {
        if (currentPlotIdx.length == series.length) {
            return currentPlotIdx
        }
        if (currentPlotIdx.length > series.length) {
            return currentPlotIdx.slice(0, series.length)
        }
        if (currentPlotIdx.length < series.length) {
            while (currentPlotIdx.length != series.length) {
                currentPlotIdx.push(-1)
            }
            return currentPlotIdx
        }
    }

    let plotIdx = []
    for (let s of series) {
        plotIdx.push(-1)
    }

    return plotIdx
}

export function toPointValue(s: IndicatorModel) {
    let res: PointValue[] = []
    let step = getListFromBinary(s.step)
    let value = getListFromBinary(s.value)
    let lastStep = getListFromBinary(s.last_step)
    for (let i = 0; i < step.length; ++i) {
        res.push({step: step[i], value: value[i], smoothed: value[i], lastStep: lastStep[i]})
    }

    return res
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

export function trimSteps(series: Indicator[], min: number, max: number, smoothRange: number = 0) : Indicator[] {
    smoothRange /= 2 // remove half from each end
    return  <Indicator[]>series.map(s => {
        let res = {...s}
        res.series = []

        if (s.series.length == 0) {
            return res
        }

        let start = 1e9, end = -1
        let localMin = min
        let localMax = max

        if (localMin == -1) {
            localMin = s.series[0].step
        }
        localMin = Math.max(localMin, s.series[0].step + smoothRange)

        if (localMax == -1) {
            localMax = s.series[s.series.length - 1].step
        }
        localMax = Math.min(localMax, s.series[s.series.length - 1].step - smoothRange)

        for (let i = 0; i < s.series.length; i++) {
            let p = s.series[i]
            if ((p.step >= localMin || localMin == -1) && (p.step <= localMax || localMax == -1)) {
                start = Math.min(start, i)
                end = Math.max(end, i)
                res.series.push(p)
            }
        }

        return res
    })
}

export function trimStepsOfPoints(series: PointValue[][], min: number, max: number, smoothRange: number = 0) : PointValue[][] {
     smoothRange /= 2 // remove half from each end
    return series.map(s => {
        let res = []
        let start = 1e9, end = -1
        let localMin = min
        let localMax = max

        if (localMin == -1) {
            localMin = s[0].step
        }
        localMin = Math.max(localMin, s[0].step + smoothRange)

        if (localMax == -1) {
            localMax = s[s.length - 1].step
        }
        localMax = Math.min(localMax, s[s.length - 1].step - smoothRange)
        for (let i = 0; i < s.length; i++) {
            let p = s[i]
            if ((p.step >= localMin || localMin == -1) && (p.step <= localMax || localMax == -1)) {
                start = Math.min(start, i)
                end = Math.max(end, i)
                res.push(p)
            }
        }

        return res
    })
}

/**
 * Calculates the smooth window size for each series in the current and base series.
 * The smooth window size is determined based on the minimum range of steps in the series and the provided smooth value.
 *
 * @param {Indicator[]} currentSeries - The current series of data.
 * @param {Indicator[]} baseSeries - The base series of data for comparison.
 * @param {number} smoothValue - The value to be used for smoothing the data.
 * @returns {[number[][], number]} - Returns an array of smooth window sizes for each series. and the smooth window size in steps.
 * (ret[0] = smooth window for current series, ret[1] = smooth window for base series
 */
export function getSmoothWindow(currentSeries: Indicator[], baseSeries: Indicator[], smoothValue: number): [number[][], number] {
    let minRange: number = Number.MAX_SAFE_INTEGER
    for (let s of currentSeries) {
        if (s.series.length > 0 && !s.is_summary) {
            minRange = Math.min(minRange, s.series[s.series.length - 1].step - s.series[0].step)
        }
    }
    for (let s of baseSeries) {
        if (s.series.length > 0 && !s.is_summary) {
            minRange = Math.min(minRange, s.series[s.series.length - 1].step - s.series[0].step)
        }
    }

    if (minRange == Number.MAX_SAFE_INTEGER) {
        let stepRange = [[],[]]
        for (let s of currentSeries) {
            stepRange[0].push(1)
        }
        for (let s of baseSeries) {
            stepRange[1].push(1)
        }
        return [stepRange, 0]
    }

    let smoothRange = mapRange(smoothValue, 1, 100, 1, minRange/10)

    let stepRange = [[],[]]
    for (let s of currentSeries) {
        if (s.series.length >= 2 && !s.is_summary) {
            let stepGap = s.series[1].step - s.series[0].step
            let numSteps = Math.max(1, Math.floor(smoothRange / stepGap))
            stepRange[0].push(numSteps)
        } else {
            stepRange[0].push(1)
        }
    }
    for (let s of baseSeries) {
        if (s.series.length >= 2 && !s.is_summary) {
            let stepGap = s.series[1].step - s.series[0].step
            let numSteps = Math.max(1, Math.floor(smoothRange / stepGap))
            stepRange[1].push(numSteps)
        } else {
            stepRange[1].push(1)
        }
    }

    return [stepRange, smoothRange]
}

// Default smoothing function from backend
function meanAngle(smoothed: PointValue[], aspectRatio: number) {
    let xRange = smoothed[smoothed.length - 1].lastStep - smoothed[0].lastStep
    let yExtent = getExtent([smoothed], d => d.smoothed)
    let yRange = yExtent[1] - yExtent[0]

    if (xRange < 1e-9 || yRange < 1e-9) {
        return 0
    }

    let angles = []
    for (let i = 1; i < smoothed.length; i++) {
        let dx = (smoothed[i].lastStep - smoothed[i - 1].lastStep) / xRange
        let dy = (smoothed[i].smoothed - smoothed[i - 1].smoothed) / yRange
        angles.push(Math.atan2(Math.abs(dy) * aspectRatio, Math.abs(dx)))
    }

    let angleSum = 0
    for (let a of angles) {
        angleSum += a
    }
    return angleSum / angles.length
}

export function smooth45(series: PointValue[]): PointValue[] {
    const MIN_SMOOTH_POINTS = 1
    let fortyFive = Math.PI / 4
    let hi = Math.max(1, series.length / MIN_SMOOTH_POINTS)
    let lo = 1

    let smoothed = smoothSeries(series, hi)

    while (lo < hi) {
        let m = (lo + hi) / 2
        smoothed = smoothSeries(series, m)
        let angle = meanAngle(smoothed, 0.5)
        if (angle > fortyFive) {
            lo = m + 1
        } else {
            hi = m
        }
    }

    return smoothSeries(series, hi)
}

function getListFromBinary(binaryString: string): Float32Array {
    let b64 = atob(binaryString)
    const uint8Array = new Uint8Array(b64.length);
    for (let i = 0; i < b64.length; i++) {
      uint8Array[i] = b64.charCodeAt(i);
    }
    try {
        let dataView = new DataView(uint8Array.buffer)
        let floatArray = new Float32Array(dataView.byteLength / 4)
        for (let i = 0; i < floatArray.length; i++) {
            floatArray[i] = dataView.getFloat32(i * 4, true)
        }
        return floatArray
    } catch (e) {
        console.error("Error parsing binary data", e)
        return  new Float32Array(0);
    }
}