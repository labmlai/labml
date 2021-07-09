import {RunStatus} from "./status"

export interface RunListItemModel {
    run_uuid: string
    computer_uuid : string
    run_status: RunStatus
    last_updated_time: number
    name: string
    comment: string
    start_time: number
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

    constructor(run_list_item: RunListItemModel) {
        this.run_uuid = run_list_item.run_uuid
        this.computer_uuid = run_list_item.computer_uuid
        this.name = run_list_item.name
        this.comment = run_list_item.comment
        this.start_time = run_list_item.start_time
        this.last_updated_time = run_list_item.last_updated_time
        this.run_status = new RunStatus(run_list_item.run_status)
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
