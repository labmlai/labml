import {AnalysisDataCache, AnalysisPreferenceCache, SessionStatusCache} from "../../../cache/cache"
import {AnalysisCache} from "../../helpers"

class DiskAnalysisCache extends AnalysisDataCache {
    constructor(uuid: string, statusCache: SessionStatusCache) {
        super(uuid, 'disk', statusCache)
    }
}

class DiskPreferenceCache extends AnalysisPreferenceCache {
    constructor(uuid: string) {
        super(uuid, 'disk')
    }
}

let diskCache = new AnalysisCache('session', DiskAnalysisCache, DiskPreferenceCache)

export default diskCache
