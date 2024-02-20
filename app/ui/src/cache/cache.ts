import {AnalysisDataModel, Run} from "../models/run"
import {Status} from "../models/status"
import NETWORK from "../network"
import { User} from "../models/user"
import {RunListItemModel, RunsList} from '../models/run_list'
import {AnalysisPreferenceModel, ComparisonPreferenceModel} from "../models/preferences"
import {SessionsList} from '../models/session_list'
import {Session} from '../models/session'

const RELOAD_TIMEOUT = 60 * 1000
const FORCE_RELOAD_TIMEOUT = 5 * 1000

export function isReloadTimeout(lastUpdated: number): boolean {
    return (new Date()).getTime() - lastUpdated > RELOAD_TIMEOUT
}

export function isForceReloadTimeout(lastUpdated: number): boolean {
    return (new Date()).getTime() - lastUpdated > FORCE_RELOAD_TIMEOUT
}

class BroadcastPromise<T> {
    // Registers a bunch of promises and broadcast to all of them
    private isLoading: boolean
    private resolvers: any[]
    private rejectors: any[]

    constructor() {
        this.isLoading = false
        this.resolvers = []
        this.rejectors = []
    }

    forceCreate(load: () => Promise<T>): Promise<T> {
        if (this.isLoading) {
            this.reject("")
            this.isLoading = false
        }
        return this.create(load)
    }

    create(load: () => Promise<T>): Promise<T> {
        let promise = new Promise<T>((resolve, reject) => {
            this.add(resolve, reject)
        })

        if (!this.isLoading) {
            this.isLoading = true
            // Load again only if not currently loading;
            // Otherwise resolve/reject will be called when the current loading completes.
            load().then((res: T) => {
                this.resolve(res)
            }).catch((err) => {
                this.reject(err)
            })
        }

        return promise
    }

    private add(resolve: (value: T) => void, reject: (err: any) => void) {
        this.resolvers.push(resolve)
        this.rejectors.push(reject)
    }

    private resolve(value: T) {
        this.isLoading = false
        let resolvers = this.resolvers
        this.resolvers = []
        this.rejectors = []

        for (let r of resolvers) {
            r(value)
        }
    }

    private reject(err: any) {
        this.isLoading = false
        let rejectors = this.rejectors
        this.resolvers = []
        this.rejectors = []

        for (let r of rejectors) {
            r(err)
        }
    }
}

export abstract class CacheObject<T> {
    public lastUpdated: number
    protected data!: T
    protected broadcastPromise = new BroadcastPromise<T>()
    private lastUsed: number

    constructor() {
        this.lastUsed = 0
        this.lastUpdated = 0
    }

    abstract load(...args: any[]): Promise<T>

    async get(isRefresh = false, ...args: any[]): Promise<T> {
        if (this.data == null || (isRefresh && isForceReloadTimeout(this.lastUpdated)) || isReloadTimeout(this.lastUpdated)) {
            this.data = await this.load()
            this.lastUpdated = (new Date()).getTime()
        }

        this.lastUsed = new Date().getTime()

        return this.data
    }

    invalidate_cache(): void {
        this.data = null
    }
}

export class RunsListCache extends CacheObject<RunsList> {
    async load(...args: any[]): Promise<RunsList> {
        return this.broadcastPromise.create(async () => {
            let res = await NETWORK.getRuns(args[0])
            return new RunsList(res)
        })
    }

    async get(isRefresh = false, ...args: any[]): Promise<RunsList> {
        if (args && args[0]) {
            return await this.load(args[0])
        }

        if (this.data == null || (isRefresh && isForceReloadTimeout(this.lastUpdated)) || isReloadTimeout(this.lastUpdated)) {
            this.data = await this.load(null)
            this.lastUpdated = (new Date()).getTime()
        }

        return this.data
    }

    async deleteRuns(runUUIDS: Array<string>): Promise<void> {
        let runs = []
        // Only updating the cache manually, if the cache exists
        if (this.data) {
            let currentRuns = this.data.runs
            for (let run of currentRuns) {
                if (!runUUIDS.includes(run.run_uuid)) {
                    runs.push(run)
                }
            }

            this.data.runs = runs
        }
        await NETWORK.deleteRuns(runUUIDS)
    }

    async addRun(run: Run): Promise<void> {
        await NETWORK.addRun(run.run_uuid)
        this.invalidate_cache()
    }

    async claimRun(run: Run): Promise<void> {
        await NETWORK.claimRun(run.run_uuid)
        this.invalidate_cache()
    }
}

export class SessionsListCache extends CacheObject<SessionsList> {
    async load(): Promise<SessionsList> {
        return this.broadcastPromise.create(async () => {
            let res = await NETWORK.getSessions()
            return new SessionsList(res)
        })
    }

    async deleteSessions(sessionUUIDS: Set<string>): Promise<void> {
        let sessions = []
        // Only updating the cache manually, if the cache exists
        if (this.data) {
            let currentSessions = this.data.sessions
            for (let session of currentSessions) {
                if (!sessionUUIDS.has(session.session_uuid)) {
                    sessions.push(session)
                }
            }

            this.data.sessions = sessions
        }
        await NETWORK.deleteSessions(Array.from(sessionUUIDS))
    }

    async addSession(session: Session): Promise<void> {
        await NETWORK.addSession(session.session_uuid)
        this.invalidate_cache()
    }

    async claimSession(session: Session): Promise<void> {
        await NETWORK.claimSession(session.session_uuid)
        this.invalidate_cache()
    }
}

export class RunCache extends CacheObject<Run> {
    private readonly uuid: string
    private statusCache: RunStatusCache

    constructor(uuid: string, statusCache: RunStatusCache) {
        super()
        this.uuid = uuid
        this.statusCache = statusCache
    }

    async load(): Promise<Run> {
        return this.broadcastPromise.create(async () => {
            let res = await NETWORK.getRun(this.uuid)
            return new Run(res)
        })
    }

    async get(isRefresh = false): Promise<Run> {
        let status = await this.statusCache.get()

        if (this.data == null || (status.isRunning && isReloadTimeout(this.lastUpdated)) || (isRefresh && isForceReloadTimeout(this.lastUpdated))) {
            this.data = await this.load()
            this.lastUpdated = (new Date()).getTime()

            if ((status.isRunning && isReloadTimeout(this.lastUpdated)) || isRefresh) {
                await this.statusCache.get(true)
            }
        }

        return this.data
    }

    async setRun(run: Run): Promise<void> {
        await NETWORK.setRun(this.uuid, run)
    }
}

export class SessionCache extends CacheObject<Session> {
    private readonly uuid: string

    constructor(uuid: string) {
        super()
        this.uuid = uuid
    }

    async load(): Promise<Session> {
        return this.broadcastPromise.create(async () => {
            let res = await NETWORK.getSession(this.uuid)
            return new Session(res)
        })
    }

    async setSession(session: Session): Promise<void> {
        await NETWORK.setSession(this.uuid, session)
    }
}

export abstract class StatusCache extends CacheObject<Status> {
}

export class RunStatusCache extends StatusCache {
    private readonly uuid: string

    constructor(uuid: string) {
        super()
        this.uuid = uuid
    }

    async load(): Promise<Status> {
        return this.broadcastPromise.create(async () => {
            let res = await NETWORK.getRunStatus(this.uuid)
            return new Status(res)
        })
    }
}

export class SessionStatusCache extends StatusCache {
    private readonly uuid: string

    constructor(uuid: string) {
        super()
        this.uuid = uuid
    }

    async load(): Promise<Status> {
        return this.broadcastPromise.create(async () => {
            let res = await NETWORK.getSessionStatus(this.uuid)
            return new Status(res)
        })
    }
}

export class UserCache extends CacheObject<User> {
    async load(): Promise<User> {
        return this.broadcastPromise.create(async () => {
            let res = await NETWORK.getUser()
            return new User(res.user)
        })
    }

    async setUser(user: User) {
        await NETWORK.setUser(user)
    }
}

export class AnalysisDataCache extends CacheObject<AnalysisDataModel> {
    private readonly uuid: string
    private readonly url: string
    private statusCache: StatusCache
    private currentUUID: string
    private readonly isExperiment: boolean
    private currentXHR: XMLHttpRequest | null

    constructor(uuid: string, url: string, statusCache: StatusCache, isExperiment: boolean = false) {
        super()
        this.uuid = uuid
        this.statusCache = statusCache
        this.url = url
        this.isExperiment = isExperiment
        this.currentXHR = null
    }

    public setCurrentUUID(currentUUID: string) {
        this.currentUUID = currentUUID
    }

    async load(): Promise<AnalysisDataModel> {
        return this.broadcastPromise.create(async () => {
            let response = NETWORK.getAnalysis(this.url, this.uuid, false, this.currentUUID, this.isExperiment)
            this.currentXHR = response.xhr
            return await response.promise
        })
    }

    async getAllMetrics(): Promise<AnalysisDataModel> {
        if (this.currentXHR != null) {
            this.currentXHR.abort()
        }
        this.data = await this.broadcastPromise.forceCreate(async () => {
            this.lastUpdated = (new Date()).getTime()
            let response = NETWORK.getAnalysis(this.url, this.uuid, true, this.currentUUID, this.isExperiment)
            return response.promise
        })
        return this.data
    }

    async get(isRefresh = false): Promise<AnalysisDataModel> {
        let status = await this.statusCache.get()

        if (this.data == null || (status.isRunning && isReloadTimeout(this.lastUpdated)) || (isRefresh && isForceReloadTimeout(this.lastUpdated))) {
            this.data = await this.load()
            this.lastUpdated = (new Date()).getTime()

            if ((status.isRunning && isReloadTimeout(this.lastUpdated)) || isRefresh) {
                await this.statusCache.get(true)
            }
        }

        return this.data
    }
}

export class AnalysisPreferenceCache extends CacheObject<AnalysisPreferenceModel> {
    private readonly uuid: string
    private readonly url: string

    constructor(uuid: string, url: string) {
        super()
        this.uuid = uuid
        this.url = url
    }

    async load(): Promise<AnalysisPreferenceModel> {
        return this.broadcastPromise.create(async () => {
            return await NETWORK.getPreferences(this.url, this.uuid)
        })
    }

    async setPreference(preference: AnalysisPreferenceModel): Promise<void> {
        this.data = preference
        await NETWORK.updatePreferences(this.url, this.uuid, preference)
    }
}

export class ComparisonAnalysisPreferenceCache extends CacheObject<ComparisonPreferenceModel> {
    private readonly uuid: string
    private readonly url: string

    constructor(uuid: string, url: string) {
        super()
        this.uuid = uuid
        this.url = url
    }

    async load(): Promise<ComparisonPreferenceModel> {
        return this.broadcastPromise.create(async () => {
            return await NETWORK.getPreferences(this.url, this.uuid)
        })
    }

    async setPreference(preference: ComparisonPreferenceModel): Promise<void> {
        this.data = structuredClone(preference)

        await NETWORK.updatePreferences(this.url, this.uuid, preference)
    }

    deleteBaseExperiment(): ComparisonPreferenceModel {
        if (this.data == null) {
            return null
        }

        this.data.base_experiment = ''
        this.data.base_series_preferences = []
        this.data.base_series_names = []

        NETWORK.updatePreferences(this.url, this.uuid, this.data).then()

        return this.data
    }

    updateBaseExperiment(run: RunListItemModel): ComparisonPreferenceModel {
        if (this.data == null) {
            return null
        }

        this.data.base_experiment = run.run_uuid
        this.data.base_series_preferences = []
        this.data.base_series_names = []
        this.data.is_base_distributed = run.world_size != 0

        NETWORK.updatePreferences(this.url, this.uuid, this.data).then()

        return this.data
    }
}

class Cache {
    private runs: { [uuid: string]: RunCache }
    private sessions: { [uuid: string]: SessionCache }
    private runStatuses: { [uuid: string]: RunStatusCache }
    private sessionStatuses: { [uuid: string]: SessionStatusCache }

    private user: UserCache | null
    private runsList: RunsListCache | null
    private sessionsList: SessionsListCache | null

    constructor() {
        this.runs = {}
        this.sessions = {}
        this.runStatuses = {}
        this.sessionStatuses = {}
        this.user = null
        this.runsList = null
        this.sessionsList = null
    }

    getRun(uuid: string) {
        if (this.runs[uuid] == null) {
            this.runs[uuid] = new RunCache(uuid, this.getRunStatus(uuid))
        }

        return this.runs[uuid]
    }

    getSession(uuid: string) {
        if (this.sessions[uuid] == null) {
            this.sessions[uuid] = new SessionCache(uuid)
        }

        return this.sessions[uuid]
    }

    getRunsList() {
        if (this.runsList == null) {
            this.runsList = new RunsListCache()
        }

        return this.runsList
    }

    getSessionsList() {
        if (this.sessionsList == null) {
            this.sessionsList = new SessionsListCache()
        }

        return this.sessionsList
    }

    getRunStatus(uuid: string) {
        if (this.runStatuses[uuid] == null) {
            this.runStatuses[uuid] = new RunStatusCache(uuid)
        }

        return this.runStatuses[uuid]
    }

    getSessionStatus(uuid: string) {
        if (this.sessionStatuses[uuid] == null) {
            this.sessionStatuses[uuid] = new SessionStatusCache(uuid)
        }

        return this.sessionStatuses[uuid]
    }

    getUser() {
        if (this.user == null) {
            this.user = new UserCache()
        }

        return this.user
    }

    invalidateCache() {
        this.runs = {}
        this.sessions = {}
        this.runStatuses = {}
        this.sessionStatuses = {}
        if (this.user != null) {
            this.user.invalidate_cache()
        }
        this.runsList = null
        this.sessionsList = null
    }
}

let CACHE = new Cache()

export default CACHE
