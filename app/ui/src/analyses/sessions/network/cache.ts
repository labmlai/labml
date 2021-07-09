import {AnalysisDataCache, AnalysisPreferenceCache, SessionStatusCache} from "../../../cache/cache"
import {AnalysisCache} from "../../helpers"

class NetworkAnalysisCache extends AnalysisDataCache {
    constructor(uuid: string, statusCache: SessionStatusCache) {
        super(uuid, 'network', statusCache)
    }
}

class NetworkPreferenceCache extends AnalysisPreferenceCache {
    constructor(uuid: string) {
        super(uuid, 'network')
    }
}

let networkCache = new AnalysisCache('session', NetworkAnalysisCache, NetworkPreferenceCache)

export default networkCache
