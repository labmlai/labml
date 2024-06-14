import {AnalysisData, CustomMetric, CustomMetricList, CustomMetricModel, Logs, LogUpdateType, Run} from "../models/run"
import {Status} from "../models/status"
import NETWORK, {ErrorResponse} from "../network"
import { User} from "../models/user"
import {RunListItemModel, RunsList} from '../models/run_list'
import {AnalysisPreferenceModel, ComparisonPreferenceModel} from "../models/preferences"
import {SessionsList} from '../models/session_list'
import {Session} from '../models/session'
import {ProcessData} from "../analyses/sessions/process/types"
import {Config} from "../models/config"

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
    protected lastUsed: number

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
        delete this.data
    }
}

export enum RunsFolder {
    DEFAULT = 'default',
    ARCHIVE = 'archive',
}

export class RunsListCache extends CacheObject<RunsList> {
    private readonly folderName: string

    constructor(folderName: string) {
        super()
        this.folderName = folderName
    }


    async load(...args: any[]): Promise<RunsList> {
        return this.broadcastPromise.create(async () => {
            let res = await NETWORK.getRuns(args[0], args[1])
            return new RunsList(res)
        })
    }

    async get(isRefresh = false, ...args: any[]): Promise<RunsList> {
        if (args && args[0] && args[1]) {
            return await this.load(args[0], args[1])
        }

        if (this.data == null || (isRefresh && isForceReloadTimeout(this.lastUpdated)) || isReloadTimeout(this.lastUpdated)) {
            this.data = await this.load(null, this.folderName)
            this.lastUpdated = (new Date()).getTime()
        }

        return this.data
    }

    async deleteRuns(runUUIDS: Array<string>): Promise<void> {
        await NETWORK.deleteRuns(runUUIDS)
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
    }

    async addRun(run: Run): Promise<void> {
        await NETWORK.addRun(run.run_uuid)
        this.invalidate_cache()
    }

    async claimRun(run: Run): Promise<void> {
        await NETWORK.claimRun(run.run_uuid)
        this.invalidate_cache()
    }

    async localUpdateRun(run: Run) {
        if (this.data == null) {
            return
        }

        for (let runItem of this.data.runs) {
            if (runItem.run_uuid == run.run_uuid) {
                runItem.name = run.name
                runItem.comment = run.comment
                runItem.favorite_configs = []
                for (let c of run.configs) {
                    if (run.favourite_configs.includes(c.name)) {
                        runItem.favorite_configs.push(new Config(c))
                    }
                }
            }
        }
    }

    getRuns(runUUIDS: Array<string>): Array<RunListItemModel> {
        let runs = []
        if (this.data) {
            for (let run of this.data.runs) {
                if (runUUIDS.includes(run.run_uuid)) {
                    runs.push(run)
                }
            }
        }
        return runs
    }

    removeRuns(runUUIDS: Array<string>): void {
        if (this.data) {
            let runs = []
            for (let run of this.data.runs) {
                if (!runUUIDS.includes(run.run_uuid)) {
                    runs.push(run)
                }
            }
            this.data.runs = runs
        }
    }

    insertRuns(runs: Array<RunListItemModel>): void {
        if (this.data) {
            this.data.runs = this.data.runs.concat(runs)
            this.data.runs.sort((a, b) => {
                return b.start_time - a.start_time
            })
        }
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

    async updateRunData(data: Record<string, any>): Promise<void> {
        await NETWORK.updateRunData(this.uuid, data)
    }

    localUpdateFolder(folder: string) {
        if (this.data != null) {
            this.data.folder = folder
        }
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

export abstract class BaseDataCache<T> extends CacheObject<T> {
    protected readonly uuid: string
    protected readonly url: string
    protected statusCache: StatusCache
    protected currentUUID: string
    protected readonly isExperiment: boolean
    protected currentXHR: XMLHttpRequest | null

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

    async load(): Promise<T> {
        return this.broadcastPromise.create(async () => {
            let response = NETWORK.getAnalysis(this.url, this.uuid, false, this.currentUUID, this.isExperiment)
            this.currentXHR = response.xhr
            return this.createInstance(await response.promise)
        })
    }

    async getAllMetrics(): Promise<T> {
        if (this.currentXHR != null) {
            this.currentXHR.abort()
        }
        this.data = await this.broadcastPromise.forceCreate(async () => {
            this.lastUpdated = (new Date()).getTime()
            let response = NETWORK.getAnalysis(this.url, this.uuid, true, this.currentUUID, this.isExperiment)
            return this.createInstance(await response.promise)
        })
        return this.data
    }

    async get(isRefresh = false): Promise<T> {
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

    protected abstract createInstance(data: any): T;
}

export class AnalysisDataCache extends BaseDataCache<AnalysisData> {
    protected createInstance(data: any): AnalysisData {
        return new AnalysisData(data);
    }
}

export class ProcessDataCache extends BaseDataCache<ProcessData> {
    protected createInstance(data: any): ProcessData {
        return new ProcessData(data);
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

    async deleteBaseExperiment(): Promise<ComparisonPreferenceModel> {
        if (this.data == null) {
            return null
        }

        this.data.base_experiment = ''
        this.data.base_series_preferences = []
        this.data.base_series_names = []

        await NETWORK.updatePreferences(this.url, this.uuid, this.data)

        return this.data
    }

    async updateBaseExperiment(run: RunListItemModel): Promise<ComparisonPreferenceModel> {
        if (this.data == null) {
            return null
        }

        this.data.base_experiment = run.run_uuid
        this.data.base_series_preferences = []
        this.data.base_series_names = []
        this.data.is_base_distributed = run.world_size != 0

        await NETWORK.updatePreferences(this.url, this.uuid, this.data)

        return this.data
    }
}

export class CustomMetricCache extends CacheObject<CustomMetricList> {
    private readonly uuid: string

    constructor(uuid: string) {
        super()
        this.uuid = uuid
    }

    async load(): Promise<CustomMetricList> {
        return this.broadcastPromise.create(async () => {
            let data = await NETWORK.getCustomMetrics(this.uuid)
            return new CustomMetricList(data)
        })
    }

    async createMetric(data: object): Promise<CustomMetric> {
        let customMetricModel = await NETWORK.createCustomMetric(this.uuid, data)
        let customMetric = new CustomMetric(customMetricModel)

        if (this.data == null) {
            this.data = new CustomMetricList({metrics: [customMetricModel]})
        }

        this.data.addMetric(customMetric)

        return customMetric
    }

    async deleteMetric(metricUUID: string): Promise<void> {
        await NETWORK.deleteCustomMetric(this.uuid, metricUUID)

        if (this.data != null) {
            this.data.removeMetric(metricUUID)
        }
    }
    
    async updateMetric(metricUUID: string, data: object): Promise<void> {
        await NETWORK.updateCustomMetric(this.uuid, data)

        if (this.data != null) {
            this.data.updateMetric(new CustomMetric(<CustomMetricModel>data))
        }
    }
}

export class LogCache extends CacheObject<Logs> {
    private readonly url: string
    private readonly uuid: string

    constructor(uuid: string, url: string) {
        super();

        this.url = url
        this.uuid = uuid
    }

    load(...args: any[]): Promise<Logs> {
        return this.broadcastPromise.create(async () => {
            let data = await NETWORK.getLogs(this.uuid, this.url, args[0])
            return new Logs(data)
        })
    }

    private async getAll(isRefresh = false): Promise<Logs> {
        if (isRefresh || this.data == null) {
            let data = new Logs(await NETWORK.getLogs(this.uuid, this.url, LogUpdateType.ALL))
            this.data.mergeLogs(data)
            return this.data
        }

        for (let pageNo = 0; pageNo < this.data.pageLength; pageNo++) {
            if (!this.data.hasPage(pageNo)) {
                let data = new Logs(await NETWORK.getLogs(this.uuid, this.url, LogUpdateType.ALL))
                this.data.mergeLogs(data)
                break
            }
        }

        return this.data
    }

    async getLast(isRefresh = false): Promise<Logs> {
        if (!isRefresh && this.data != null && this.data.hasPage(this.data.pageLength - 1)) {
            return this.data.getPageAsLog(this.data.pageLength - 1)
        }

        await this.get(isRefresh)
        return this.data
    }

    async getPage(pageNo: number, isRefresh = false): Promise<Logs> {
        if (pageNo == -2) {
            return await this.getAll(isRefresh)
        }

        await this.get(false)

        if (!isRefresh && this.data.hasPage(pageNo)) {
            return this.data.getPageAsLog(pageNo)
        }

        let data = new Logs(await NETWORK.getLogs(this.uuid, this.url, pageNo))
        this.data.mergeLogs(data)
        return data
    }

    async get(isRefresh = false, ...args: any[]): Promise<Logs> {
        if (this.data == null || (isRefresh && isForceReloadTimeout(this.lastUpdated)) || isReloadTimeout(this.lastUpdated)) {
            this.data = await this.load(LogUpdateType.LAST)

            this.lastUpdated = (new Date()).getTime()
        }

        this.lastUsed = new Date().getTime()

        return this.data
    }
}

class Cache {
    private runs: { [uuid: string]: RunCache }
    private customMetrics: { [uuid: string]: CustomMetricCache }
    private sessions: { [uuid: string]: SessionCache }
    private runStatuses: { [uuid: string]: RunStatusCache }
    private sessionStatuses: { [uuid: string]: SessionStatusCache }

    private user: UserCache | null
    private runsList: { [folderName: string]: RunsListCache }
    private sessionsList: SessionsListCache | null

    constructor() {
        this.runs = {}
        this.sessions = {}
        this.runStatuses = {}
        this.sessionStatuses = {}
        this.customMetrics = {}
        this.runsList = {}
        this.user = null
        this.sessionsList = null
    }

    getCustomMetrics(uuid: string) {
        if (this.customMetrics[uuid] == null) {
            this.customMetrics[uuid] = new CustomMetricCache(uuid)
        }

        return this.customMetrics[uuid]
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

    getRunsList(folderName: string) {
        if (this.runsList[folderName] == null) {
            this.runsList[folderName] = new RunsListCache(folderName)
        }

        return this.runsList[folderName]
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

    async archiveRuns(runUUIDS: Array<string>): Promise<ErrorResponse> {
        let response: ErrorResponse = await NETWORK.archiveRuns(runUUIDS)
        if (response.is_successful == false) {
            return response
        }

        let defaultFolder = this.getRunsList(RunsFolder.DEFAULT)
        let archiveFolder = this.getRunsList(RunsFolder.ARCHIVE)
        let runs = defaultFolder.getRuns(runUUIDS)
        archiveFolder.insertRuns(runs)
        defaultFolder.removeRuns(runUUIDS)

        for (let uuid of runUUIDS) {
            let run = this.getRun(uuid)
            run.localUpdateFolder(RunsFolder.ARCHIVE)
        }

        return response
    }

    async unarchiveRuns(runUUIDS: Array<string>): Promise<ErrorResponse> {
        let response: ErrorResponse = await NETWORK.unarchiveRuns(runUUIDS)
        if (response.is_successful == false) {
            return response
        }

        let defaultFolder = this.getRunsList(RunsFolder.DEFAULT)
        let archiveFolder = this.getRunsList(RunsFolder.ARCHIVE)
        let runs = archiveFolder.getRuns(runUUIDS)
        defaultFolder.insertRuns(runs)
        archiveFolder.removeRuns(runUUIDS)

        for (let uuid of runUUIDS) {
            let run = this.getRun(uuid)
            run.localUpdateFolder(RunsFolder.DEFAULT)
        }

        return response
    }

    invalidateCache() {
        this.runs = {}
        this.sessions = {}
        this.runStatuses = {}
        this.sessionStatuses = {}
        this.customMetrics = {}
        if (this.user != null) {
            this.user.invalidate_cache()
        }
        this.runsList = null
        this.sessionsList = null
    }
}

let CACHE = new Cache()

export default CACHE
