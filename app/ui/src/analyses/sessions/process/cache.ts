import {
    AnalysisPreferenceCache,
    ProcessDataCache,
    SessionStatusCache
} from "../../../cache/cache"
import {AnalysisCache} from "../../helpers"
import {DetailsCache, DetailsDataCache} from "./cache_helper"
class ProcessAnalysisCache extends ProcessDataCache {
    constructor(uuid: string, statusCache: SessionStatusCache) {
        super(uuid, 'process', statusCache)
    }
}

class ProcessPreferenceCache extends AnalysisPreferenceCache {
    constructor(uuid: string) {
        super(uuid, 'process')
    }
}

let processCache = new AnalysisCache('session', ProcessAnalysisCache, ProcessPreferenceCache)

class ProcessDetailsCache extends DetailsDataCache {
    constructor(uuid: string, processId: string, statusCache: SessionStatusCache) {
        super(uuid, processId, statusCache)
    }
}

let processDetailsCache = new DetailsCache(ProcessDetailsCache)

export {
    processCache,
    processDetailsCache
}
