import {Analysis} from "./types"

import metricAnalysis from "./experiments/metrics"
import stdOutAnalysis from "./experiments/stdout"
import stderrAnalysis from "./experiments/stderror"
import loggerAnalysis from "./experiments/logger"
import runConfigsAnalysis from "./experiments/configs"

import cpuAnalysis from './sessions/cpu'
import diskAnalysis from './sessions/disk'
import {gpuMemoryAnalysis, gpuPowerAnalysis, gpuTempAnalysis, gpuUtilAnalysis} from './sessions/gpu'
import memoryAnalysis from './sessions/memory'
import networkAnalysis from './sessions/network'
import processAnalysis from './sessions/process'
import batteryAnalysis from './sessions/battery'
import sessionConfigsAnalysis from "./sessions/configs"
import comparisonAnalysis from './experiments/comparison'
import mergedMetricsAnalysis from "./experiments/merged_metrics"

let experimentAnalyses: Analysis[] = [
    metricAnalysis,
    comparisonAnalysis,
    stdOutAnalysis,
    stderrAnalysis,
    runConfigsAnalysis,
    loggerAnalysis
]

let distributedAnalyses: Analysis[] = [
    mergedMetricsAnalysis,
    // metricAnalysis,
    comparisonAnalysis,
    stdOutAnalysis,
    stderrAnalysis,
    runConfigsAnalysis,
    loggerAnalysis
]

let rankAnalysis: Analysis[] = [
    metricAnalysis,
    stdOutAnalysis,
    stderrAnalysis,
    runConfigsAnalysis,
    loggerAnalysis
]

let sessionAnalyses: Analysis[] = [
    cpuAnalysis,
    processAnalysis,
    memoryAnalysis,
    diskAnalysis,
    gpuUtilAnalysis,
    gpuTempAnalysis,
    gpuMemoryAnalysis,
    gpuPowerAnalysis,
    batteryAnalysis,
    networkAnalysis,
    sessionConfigsAnalysis,
]

export {
    experimentAnalyses,
    sessionAnalyses,
    distributedAnalyses,
    rankAnalysis
}
