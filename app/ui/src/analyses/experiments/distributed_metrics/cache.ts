import {
    RunStatusCache,
    AnalysisDataCache,
    DistAnalysisPreferenceCache,
} from "../../../cache/cache"

export class DistMetricsAnalysisCache extends AnalysisDataCache {
    constructor(uuid: string, statusCache: RunStatusCache) {
        super(uuid, 'distributed/metrics', statusCache)
    }
}

export class DistMetricsPreferenceCache extends DistAnalysisPreferenceCache {
    constructor(uuid: string) {
        super(uuid, 'distributed/metrics')
    }
}
