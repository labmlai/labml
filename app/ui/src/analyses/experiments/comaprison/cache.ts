import {
    RunStatusCache,
    AnalysisDataCache,
    AnalysisPreferenceCache,
    ComparisonAnalysisPreferenceCache
} from "../../../cache/cache"
import {AnalysisCache} from "../../helpers"

class ComparisonAnalysisCache extends AnalysisDataCache {
    constructor(uuid: string, statusCache: RunStatusCache) {
        super(uuid, 'compare/metrics', statusCache, true)
    }
}

class ComparisonPreferenceCache extends ComparisonAnalysisPreferenceCache {
    constructor(uuid: string) {
        super(uuid, 'compare')
    }
}

let comparisonCache = new AnalysisCache('run', ComparisonAnalysisCache, ComparisonPreferenceCache)

export default comparisonCache
