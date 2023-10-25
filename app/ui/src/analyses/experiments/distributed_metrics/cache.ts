import {
    RunStatusCache,
    AnalysisDataCache,
    DistAnalysisPreferenceCache,
} from "../../../cache/cache"
import {DistAnalysisPreferenceModel} from "../../../models/preferences";

export class DistMetricsAnalysisCache extends AnalysisDataCache {
    constructor(uuid: string, statusCache: RunStatusCache) {
        super(uuid, 'distributed/metrics', statusCache)
    }
}

export class DistMetricsPreferenceCache extends DistAnalysisPreferenceCache {
    private readonly worldSize: number
    private readonly seriesSize: number

    constructor(uuid: string, worldSize: number, seriesSize: number) {
        super(uuid, 'distributed/metrics')
        this.worldSize = worldSize
        this.seriesSize = seriesSize
    }

    async get(force: boolean): Promise<DistAnalysisPreferenceModel> {
        let data: DistAnalysisPreferenceModel = await super.get(force)
        if (data == null) {
            return data
        }

        while (data.series_preferences.length < this.worldSize) {
            data.series_preferences.push([])
        }

        for (let i = 0; i < this.worldSize; i++) {
            while (data.series_preferences[i].length < this.seriesSize) {
                data.series_preferences[i].push(-1)
            }
        }

        return data
    }
}
