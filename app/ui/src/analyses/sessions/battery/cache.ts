import {AnalysisDataCache, AnalysisPreferenceCache, SessionStatusCache} from "../../../cache/cache"
import {AnalysisCache} from "../../helpers"

class BatteryAnalysisCache extends AnalysisDataCache {
    constructor(uuid: string, statusCache: SessionStatusCache) {
        super(uuid, 'battery', statusCache)
    }
}

class BatteryPreferenceCache extends AnalysisPreferenceCache {
    constructor(uuid: string) {
        super(uuid, 'battery')
    }
}

let batteryCache = new AnalysisCache('session', BatteryAnalysisCache, BatteryPreferenceCache)

export default batteryCache
