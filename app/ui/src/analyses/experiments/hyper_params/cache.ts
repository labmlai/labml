import {RunStatusCache, AnalysisDataCache, AnalysisPreferenceCache} from "../../../cache/cache"
import {AnalysisCache} from "../../helpers"

class HyperPramsAnalysisCache extends AnalysisDataCache {
    constructor(uuid: string, statusCache: RunStatusCache) {
        super(uuid, 'hyper_params', statusCache)
    }
}

class HyperPramsPreferenceCache extends AnalysisPreferenceCache {
    constructor(uuid: string) {
        super(uuid, 'hyper_params')
    }
}

let hyperParamsCache = new AnalysisCache('run', HyperPramsAnalysisCache, HyperPramsPreferenceCache)

export default hyperParamsCache
