import {
    CacheObject,
    isForceReloadTimeout,
    isReloadTimeout,
} from "../../../cache/cache"
import {Indicator, IndicatorModel} from "../../../models/run"
import NETWORK from "../../../network";

interface MetricDataModel {
    series: IndicatorModel[]
}

class MetricData {
    series: Indicator[]

    constructor(data: MetricDataModel | Indicator[]) {
        if (Array.isArray(data)) {
            this.series = data
        } else {
            this.series = []
            for (let s of data.series) {
                this.series.push(new Indicator(s))
            }
        }
    }

    public merge(data: MetricData) {
        for (let series of data.series) {
            let index = this.series.findIndex(s => s.name === series.name)
            if (index === -1) {
                this.series.push(series)
            } else {
                if (this.series[index].is_summary) { // update anyway
                    this.series[index] = series
                } else {
                    if (this.series[index].series[this.series[index].series.length - 1].step
                        == series.series[series.series.length - 1].step) {
                        continue // got a summary but have an updated series locally
                    }

                    this.series[index] = series
                }
            }
        }
    }
}

export class MetricCache extends CacheObject<MetricData> {
    private baseUrl = 'compare/metrics'
    private readonly uuid: string
    private currentXHR: XMLHttpRequest | null

    constructor(uuid: string) {
        super()
        this.uuid = uuid
    }

    async load(args: any): Promise<MetricData> {
        let response = NETWORK.getAnalysis(this.baseUrl, this.uuid, args)

        this.currentXHR = response.xhr
        let data = await response.promise
        this.currentXHR = null

        return new MetricData(data)
    }

    async get(isRefresh = false, ...args: any[]): Promise<MetricData> {
        if (this.data == null || !this.checkDataExists(args[0]) ||
            (isRefresh && isForceReloadTimeout(this.lastUpdated)) || isReloadTimeout(this.lastUpdated)) {

            let data = await this.load({
                indicators: args[0]
            })

            if (this.data != null) {
                this.data.merge(data)
            } else {
                this.data = data
            }

            this.lastUpdated = (new Date()).getTime()
        }

        this.lastUsed = new Date().getTime()

        let indicators = []
        for (let s of this.data.series) {
            indicators.push(s.getCopy()) // deep copy
        }

        for (let series of indicators) {
            if (!args[0].includes(series.name)) {
                series.is_summary = true
            }
        }

        return new MetricData(indicators)
    }

    private cancel() {
        if (this.currentXHR != null) {
            this.currentXHR.abort()
        }
    }

    private checkDataExists(indicators: string[]) {
        if (this.data == null) {
            return false
        }

        for (let series of this.data.series) {
            if (series.is_summary && indicators.includes(series.name)) {
                return false
            }
            indicators = indicators.filter(indicator => indicator !== series.name)
        }

        return indicators.length == 0
    }
}

export class MetricAnalysisCache {
    private readonly seriesCaches: { [uuid: string]: MetricCache }

    constructor() {
        this.seriesCaches = {}
    }

    getAnalysis(uuid: string) {
        if (this.seriesCaches[uuid] == null) {
            this.seriesCaches[uuid] = new MetricCache(uuid)
        }

        return this.seriesCaches[uuid]
    }
}

let metricsCache = new MetricAnalysisCache()
export default metricsCache
