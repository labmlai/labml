import {AnalysisDataCache, AnalysisPreferenceCache, SessionStatusCache} from "../../../cache/cache"
import {AnalysisCache} from "../../helpers"

class CPUAnalysisCache extends AnalysisDataCache {
    constructor(uuid: string, statusCache: SessionStatusCache) {
        super(uuid, 'cpu', statusCache)
    }
}

class CPUPreferenceCache extends AnalysisPreferenceCache {
    constructor(uuid: string) {
        super(uuid, 'cpu')
    }
}

let cpuCache = new AnalysisCache('session', CPUAnalysisCache, CPUPreferenceCache)

export default cpuCache
