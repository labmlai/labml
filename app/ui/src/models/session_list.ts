import {RunStatus} from "./status"

export interface SessionsListItemModel {
    computer_uuid: string
    session_uuid: string
    run_status: RunStatus
    last_updated_time: number
    name: string
    comment: string
    start_time: number
}

export interface SessionsListModel {
    sessions: SessionsListItemModel[]
    labml_token: string
}

export class SessionListItem {
    computer_uuid: string
    session_uuid: string
    run_status: RunStatus
    last_updated_time: number
    name: string
    comment: string
    start_time: number

    constructor(sessionListItem: SessionsListItemModel) {
        this.computer_uuid = sessionListItem.computer_uuid
        this.session_uuid = sessionListItem.session_uuid
        this.name = sessionListItem.name
        this.comment = sessionListItem.comment
        this.start_time = sessionListItem.start_time
        this.last_updated_time = sessionListItem.last_updated_time
        this.run_status = new RunStatus(sessionListItem.run_status)
    }
}

export class SessionsList {
    sessions: SessionsListItemModel[]
    labml_token: string

    constructor(sessionsList: SessionsListModel) {
        this.sessions = []
        for (let c of sessionsList.sessions) {
            this.sessions.push(new SessionListItem(c))
        }
        this.labml_token = sessionsList.labml_token
    }
}
