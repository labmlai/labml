import {LogCache} from "../../../cache/cache"
import {LogAnalysisCache} from "../../helpers"

class StdLoggerCache extends LogCache {
    constructor(uuid: string) {
        super(uuid, 'std_logger');
    }
}

let stdLoggerCache = new LogAnalysisCache(StdLoggerCache)

export default stdLoggerCache