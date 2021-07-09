import {Config, ConfigModel} from "./config"

export interface SessionModel {
    computer_uuid: string
    session_uuid: string
    name: string
    comment: string
    start_time: number
    is_claimed: boolean
    is_project_session: boolean
    configs: ConfigModel[]
}

export class Session {
    computer_uuid: string
    session_uuid: string
    name: string
    comment: string
    is_claimed: boolean
    is_project_session: boolean
    start_time: number
    configs: Config[]

    constructor(session: SessionModel) {
        this.computer_uuid = session.computer_uuid
        this.session_uuid = session.session_uuid
        this.name = session.name
        this.comment = session.comment
        this.start_time = session.start_time
        this.is_claimed = session.is_claimed
        this.is_project_session = session.is_project_session
        this.configs = []
        for (let c of session.configs) {
            this.configs.push(new Config(c))
        }
    }
}


