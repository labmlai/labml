import {LogCache} from "../../../cache/cache"
import {LogAnalysisCache} from "../../helpers";

class StdOutCache extends LogCache {
    constructor(uuid: string) {
        super(uuid, 'stdout');
    }
}

let stdOutCache = new LogAnalysisCache(StdOutCache)

export default stdOutCache