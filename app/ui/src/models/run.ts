import {Config, ConfigModel} from "./config"
import {toPointValue} from "../components/charts/utils"
import {AnalysisPreferenceModel} from "./preferences";

export interface RunModel {
    run_uuid: string
    rank: number
    world_size: number
    other_rank_run_uuids: { [uuid: number]: string }
    name: string
    comment: string
    tags: string[]
    note: string
    start_time: number
    python_file: string
    repo_remotes: string
    commit: string
    commit_message: string
    start_step: number
    is_claimed: boolean
    is_project_run: boolean
    size: number
    size_checkpoints: number
    size_tensorboard: number
    computer_uuid: string
    configs: ConfigModel[]
    selected_configs: string[]
    favourite_configs: string[]
    session_id: string
    process_id: string
    folder: string
}

export interface PointValue {
    step: number
    value: number
    smoothed: number
    lastStep: number
}

export interface IndicatorModel {
    name: string
    step: string
    value: string
    mean: number
    is_summary: boolean
    last_step: string
}

export class Indicator {
    name : string
    series: PointValue[]
    is_summary: boolean
    lowTrimIndex: number
    highTrimIndex: number

    constructor(indicator: IndicatorModel) {
        this.name = indicator.name
        this.series = toPointValue(indicator)
        this.is_summary = indicator.is_summary
        this.lowTrimIndex = 0
        this.highTrimIndex = this.series.length - 1
    }

    get trimmedSeries(): PointValue[] {
        return this.series.slice(this.lowTrimIndex, this.highTrimIndex + 1)
    }

    getCopy(): Indicator {
        let copy = new Indicator({
            name: this.name,
            step: "",
            value: "",
            mean: 0,
            is_summary: this.is_summary,
            last_step: ""
        })
        copy.series = [...this.series]
        return copy
    }
}

export interface AnalysisDataModel {
    series: IndicatorModel[]
    summary: IndicatorModel[]
}

export class AnalysisData {
    series: Indicator[]
    summary: Indicator[]

    constructor(data: AnalysisDataModel) {
        this.series = []
        for (let s of data.series) {
            this.series.push(new Indicator(s))
        }
        this.summary = []
        if (data.summary != null) {
            for (let s of data.summary) {
                this.summary.push(new Indicator(s))
            }
        }
    }
}

export class Run {
    run_uuid: string
    rank: number
    world_size: number
    other_rank_run_uuids: { [uuid: number]: string }
    name: string
    comment: string
    note: string
    tags: string[]
    start_time: number
    python_file: string
    repo_remotes: string
    commit: string
    commit_message: string
    start_step: number
    is_claimed: boolean
    is_project_run: boolean
    size: number
    size_checkpoints: number
    size_tensorboard: number
    computer_uuid: string
    configs: Config[]
    dynamic: object
    selected_configs: string[]
    favourite_configs: string[]
    session_id: string
    process_id: string
    folder: string

    constructor(run: RunModel) {
        this.run_uuid = run.run_uuid
        this.rank = run.rank
        this.world_size = run.world_size
        this.other_rank_run_uuids = run.other_rank_run_uuids
        this.name = run.name
        this.comment = run.comment
        this.note = run.note
        this.tags = run.tags
        this.start_time = run.start_time
        this.python_file = run.python_file
        this.repo_remotes = run.repo_remotes
        this.commit = run.commit
        this.commit_message = run.commit_message
        this.start_step = run.start_step
        this.is_claimed = run.is_claimed
        this.is_project_run = run.is_project_run
        this.size = run.size
        this.size_checkpoints = run.size_checkpoints
        this.size_tensorboard = run.size_tensorboard
        this.computer_uuid = run.computer_uuid
        this.configs = []
        this.favourite_configs = run.favourite_configs ?? []
        this.selected_configs = run.selected_configs ?? []
        for (let c of run.configs) {
            this.configs.push(new Config(c,this.selected_configs.includes(c.key), this.favourite_configs.includes(c.key)))
        }
        this.session_id = run.session_id
        this.process_id = run.process_id
        this.folder = run.folder
    }

    public updateConfigs() {
        let updatedConfigs: Config[] = []
        for (let c of this.configs) {
            updatedConfigs.push(new Config(c,this.selected_configs.includes(c.key), this.favourite_configs.includes(c.key)))
        }
        this.configs = updatedConfigs
    }

    public get favouriteConfigs(): Config[] {
        let configs: Config[] = []
        for (let c of this.configs) {
            if (c.isFavourite) {
                configs.push(c)
            }
        }
        return configs
    }
}


export interface CustomMetricModel {
    id: string
    name: string
    description: string
    created_time: number
    preferences: AnalysisPreferenceModel
}

export class CustomMetric {
    id: string
    name: string
    description: string
    createdTime: number
    preferences: AnalysisPreferenceModel

    constructor(metric: CustomMetricModel) {
        this.id = metric.id
        this.name = metric.name
        this.description = metric.description
        this.createdTime = metric.created_time
        this.preferences = metric.preferences
    }

    public toData(): CustomMetricModel {
        return {
            id: this.id,
            name: this.name,
            description: this.description,
            created_time: this.createdTime,
            preferences: this.preferences
        }
    }
}

export interface CustomMetricListModel {
    metrics: CustomMetricModel[]
}

export class CustomMetricList {
    private readonly metrics: Record<string, CustomMetric>

    constructor(metricList: CustomMetricListModel) {
        this.metrics = {}
        for (let item of metricList.metrics) {
            this.metrics[item.id] =  new CustomMetric(item)
        }
    }

    public getMetric(metricId: string): CustomMetric {
        return this.metrics[metricId]
    }

    public getMetrics(): CustomMetric[] {
        return Object.values(this.metrics).sort((a, b) => b.createdTime - a.createdTime);
    }

    public addMetric(metric: CustomMetric) {
        this.metrics[metric.id] = metric
    }

    public removeMetric(metricId: string) {
        delete this.metrics[metricId]
    }

    public updateMetric(metric: CustomMetric) {
        this.metrics[metric.id] = metric
    }
}

export interface LogModel {
    pages: Record<string, string>
    page_length: number
}

export interface LogUpdateFeedback {
    lowerIndexes: string[]
    higherIndexes: string[]
}

export class Logs {
    pages: Record<string, string>
    pageLength: number

    constructor(logs: LogModel) {
        this.pages = logs.pages
        this.pageLength = logs.page_length
    }

    public getPage(index: number): string | null {
        return this.pages[index]
    }

    public getPageAsLog(index: number): Logs {
        let pages = {}
        pages[index] = this.getPage(index)
        return new Logs({pages: pages, page_length: this.pageLength})
    }

    public hasPage(index: number): Boolean {
        return this.pages.hasOwnProperty(index)
    }

    public mergeLogs(logs: Logs) {
        this.pages = {...this.pages, ...logs.pages}
        this.pageLength = logs.pageLength
    }
}