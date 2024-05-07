import {LogCache} from "../../../cache/cache"
import {LogAnalysisCache} from "../../helpers"

class StdErrCache extends LogCache {
    constructor(uuid: string) {
        super(uuid, 'stderr');
    }
}

let stdErrCache = new LogAnalysisCache(StdErrCache)

export default stdErrCache