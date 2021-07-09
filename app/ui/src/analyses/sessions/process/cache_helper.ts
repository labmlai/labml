import NETWORK from "../../../network"
import CACHE, {CacheObject, isReloadTimeout, SessionStatusCache, StatusCache} from "../../../cache/cache"
import {ProcessDetailsModel} from "./types"

export class DetailsDataCache extends CacheObject<ProcessDetailsModel> {
    private readonly uuid: string
    private readonly processId: string
    private statusCache: StatusCache

    constructor(uuid: string, processId: string, statusCache: StatusCache) {
        super()
        this.uuid = uuid
        this.processId = processId
        this.statusCache = statusCache
    }

    async load(): Promise<ProcessDetailsModel> {
        return this.broadcastPromise.create(async () => {
            return await NETWORK.getCustomAnalysis(`process/${this.uuid}/details/${this.processId}`)
        })
    }

    async get(isRefresh = false): Promise<ProcessDetailsModel> {
        let status = await this.statusCache.get()

        if (this.data == null || (status.isRunning && isReloadTimeout(this.lastUpdated)) || isRefresh) {
            this.data = await this.load()
            this.lastUpdated = (new Date()).getTime()

            if ((status.isRunning && isReloadTimeout(this.lastUpdated)) || isRefresh) {
                await this.statusCache.get(true)
            }
        }

        return this.data
    }
}

export class DetailsCache<TA extends DetailsDataCache> {
    private readonly series: new (uuid: string, processId: string, status: SessionStatusCache) => TA
    private readonly processCache: { [uuid: string]: { [processId: string]: DetailsDataCache } }

    constructor(series: new (uuid: string, processId: string, status: SessionStatusCache) => TA) {
        this.processCache = {}
        this.series = series
    }

    getAnalysis(uuid: string, processId: string) {
        if (this.processCache[uuid] == null) {
            let seriesCaches = {}
            seriesCaches[processId] = new this.series(uuid, processId, CACHE.getSessionStatus(uuid))
            this.processCache[uuid] = seriesCaches
        } else if (this.processCache[uuid][processId] == null) {
            this.processCache[uuid][processId] = new this.series(uuid, processId, CACHE.getSessionStatus(uuid))
        }

        return this.processCache[uuid][processId]
    }
}
