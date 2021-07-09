import {AnalysisDataCache, AnalysisPreferenceCache, SessionStatusCache} from "../../../cache/cache"
import {AnalysisCache} from "../../helpers"

class GPUAnalysisCache extends AnalysisDataCache {
    constructor(uuid: string, statusCache: SessionStatusCache) {
        super(uuid, 'gpu', statusCache)
    }
}

class GPUPreferenceCache extends AnalysisPreferenceCache {
    constructor(uuid: string) {
        super(uuid, 'gpu')
    }
}

let gpuCache = new AnalysisCache('session', GPUAnalysisCache, GPUPreferenceCache)

export default gpuCache
