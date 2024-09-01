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

    constructor(data: MetricDataModel) {
        this.series = []
        for (let s of data.series) {
            this.series.push(new Indicator(s))
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

    load(args: any): Promise<MetricData> {
        return this.broadcastPromise.create(async () => {
            let response =
                NETWORK.getAnalysis(this.baseUrl, this.uuid, args)

            this.currentXHR = response.xhr
            let data = await response.promise
            this.currentXHR = null

            return new MetricData(data)
        })
    }

    async get(isRefresh = false, ...args: any[]): Promise<MetricData> {
        if (this.data == null || !this.checkDataExists(args[0]) ||
            (isRefresh && isForceReloadTimeout(this.lastUpdated)) || isReloadTimeout(this.lastUpdated)) {
            this.cancel()

            this.data = await this.load({
                indicators: args[0]
            })
            this.lastUpdated = (new Date()).getTime()
        }

        this.lastUsed = new Date().getTime()

        return this.data
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
