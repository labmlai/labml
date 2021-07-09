import {RunStatusCache, AnalysisDataCache, AnalysisPreferenceCache} from "../../../cache/cache"
import {AnalysisCache} from "../../helpers"

class ParameterAnalysisCache extends AnalysisDataCache {
    constructor(uuid: string, statusCache: RunStatusCache) {
        super(uuid, 'parameters', statusCache)
    }
}

class ParameterPreferenceCache extends AnalysisPreferenceCache {
    constructor(uuid: string) {
        super(uuid, 'parameters')
    }
}

let parametersCache = new AnalysisCache('run', ParameterAnalysisCache, ParameterPreferenceCache)

export default parametersCache
