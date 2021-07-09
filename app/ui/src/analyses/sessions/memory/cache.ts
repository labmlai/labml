import {AnalysisDataCache, AnalysisPreferenceCache, SessionStatusCache} from "../../../cache/cache"
import {AnalysisCache} from "../../helpers"

class MemoryAnalysisCache extends AnalysisDataCache {
    constructor(uuid: string, statusCache: SessionStatusCache) {
        super(uuid, 'memory', statusCache)
    }
}

class MemoryPreferenceCache extends AnalysisPreferenceCache {
    constructor(uuid: string) {
        super(uuid, 'memory')
    }
}

let memoryCache = new AnalysisCache('session', MemoryAnalysisCache, MemoryPreferenceCache)

export default memoryCache
