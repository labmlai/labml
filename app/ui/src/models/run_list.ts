import {RunStatus} from "./status"
import {Config, ConfigModel} from "./config"

export interface MetricValue {
    name: string
    value: number
}

export interface RunListItemModel {
    run_uuid: string
    computer_uuid : string
    run_status: RunStatus
    last_updated_time: number
    name: string
    comment: string
    start_time: number
    world_size: number
    favorite_configs?: ConfigModel[]
    metric_values?: MetricValue[]
    step: number
    tags: string[]
}

export interface RunsListModel {
    runs: RunListItemModel[]
    labml_token: string
}

export class RunListItem {
    run_uuid: string
    computer_uuid : string
    run_status: RunStatus
    last_updated_time: number
    name: string
    comment: string
    start_time: number
    world_size: number
    favorite_configs?: Config[]
    metric_values?: MetricValue[]
    step: number
    tags: string[]

    constructor(run_list_item: RunListItemModel) {
        this.run_uuid = run_list_item.run_uuid
        this.computer_uuid = run_list_item.computer_uuid
        this.name = run_list_item.name
        this.comment = run_list_item.comment
        this.start_time = run_list_item.start_time
        this.last_updated_time = run_list_item.last_updated_time
        this.run_status = new RunStatus(run_list_item.run_status)
        this.world_size = run_list_item.world_size
        this.favorite_configs = []
        this.metric_values = run_list_item.metric_values
        this.step = run_list_item.step
        this.tags = run_list_item.tags.sort()
        if (run_list_item.favorite_configs != null) {
            for (let c of run_list_item.favorite_configs) {
                this.favorite_configs.push(new Config(c))
            }
        }
    }

    get mainUUID() {
        return this.run_uuid.split('_')[0]
    }
}

export class RunsList {
    runs: RunListItemModel[]
    labml_token: string

    constructor(runs_list: RunsListModel) {
        this.runs = []
        for (let r of runs_list.runs) {
            this.runs.push(new RunListItem(r))
        }
        this.labml_token = runs_list.labml_token
    }
}
