import {RunStatusCache, AnalysisDataCache, AnalysisPreferenceCache} from "../../../cache/cache"
import {AnalysisCache} from "../../helpers"

class MetricsAnalysisCache extends AnalysisDataCache {
    constructor(uuid: string, statusCache: RunStatusCache) {
        super(uuid, 'metrics', statusCache)
    }
}

class MetricsPreferenceCache extends AnalysisPreferenceCache {
    constructor(uuid: string) {
        super(uuid, 'metrics')
    }
}

let metricsCache = new AnalysisCache('run', MetricsAnalysisCache, MetricsPreferenceCache)

export default metricsCache
