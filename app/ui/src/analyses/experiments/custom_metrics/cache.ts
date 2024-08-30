import CACHE, {AnalysisDataCache, RunStatusCache} from "../../../cache/cache"

class MetricCache extends AnalysisDataCache {
    constructor(uuid: string, statusCache: RunStatusCache) {
        super(uuid, 'compare/metrics', statusCache, true)
    }
}

export class MetricAnalysisCache {
    private readonly seriesCaches: { [uuid: string]: MetricCache }

    constructor() {
        this.seriesCaches = {}
    }

    getAnalysis(uuid: string) {
        if (this.seriesCaches[uuid] == null) {
            this.seriesCaches[uuid] = new MetricCache(uuid, this.getStatus(uuid))
        }

        return this.seriesCaches[uuid]
    }

    private getStatus(uuid: string) {
        return CACHE.getRunStatus(uuid)
    }
}

let metricsCache = new MetricAnalysisCache()
export default metricsCache
