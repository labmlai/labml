import CACHE, {
    AnalysisPreferenceCache, BaseDataCache,
    ComparisonAnalysisPreferenceCache,
    RunStatusCache,
    SessionStatusCache
} from "../cache/cache"
import {ContentType} from '../types'

export class AnalysisCache<TA extends BaseDataCache<any>, TAP extends AnalysisPreferenceCache | ComparisonAnalysisPreferenceCache> {
    private readonly type: ContentType
    private readonly series: new (uuid: string, status: RunStatusCache | SessionStatusCache) => TA
    private readonly seriesCaches: { [uuid: string]: TA }
    private readonly preferences: new (uuid: string) => TAP
    private readonly preferencesCaches: { [uuid: string]: AnalysisPreferenceCache | ComparisonAnalysisPreferenceCache }

    constructor(type: ContentType, series: new (uuid: string, status: RunStatusCache | SessionStatusCache) => TA, preferences: new (uuid: string) => TAP) {
        this.type = type
        this.seriesCaches = {}
        this.preferencesCaches = {}
        this.series = series
        this.preferences = preferences
    }

    getAnalysis(uuid: string) {
        if (this.seriesCaches[uuid] == null) {
            this.seriesCaches[uuid] = new this.series(uuid, this.getStatus(uuid))
        }

        return this.seriesCaches[uuid]
    }

    getPreferences(uuid: string) {
        if (this.preferencesCaches[uuid] == null) {
            this.preferencesCaches[uuid] = new this.preferences(uuid)
        }

        return this.preferencesCaches[uuid]
    }

    private getStatus(uuid: string) {
        if (this.type === 'run') {
            return CACHE.getRunStatus(uuid)
        } else if (this.type === 'session') {
            return CACHE.getSessionStatus(uuid)
        }

        return null
    }
}
