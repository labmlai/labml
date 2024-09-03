export interface RunStatusModel {
    status: string
    details: string
    time: number
}

export interface StatusModel {
    run_uuid: string
    last_updated_time: number
    run_status: RunStatusModel
}

export class RunStatus {
    status: string
    details: string
    time: number

    constructor(runStatus: RunStatusModel) {
        this.status = runStatus.status
        this.details = runStatus.details
        this.time = runStatus.time
    }
}

export class Status {
    uuid: string
    last_updated_time: number
    run_status: RunStatus

    constructor(status: StatusModel) {
        this.uuid = status.run_uuid
        this.last_updated_time = status.last_updated_time
        this.run_status = new RunStatus(status.run_status)
    }

    get isRunning() {
        if (this.run_status.status === 'in progress') {
            let timeDiff = (Date.now() / 1000 - this.last_updated_time) / 60
            return timeDiff <= 15
        } else {
            return false
        }
    }

    get isStatusInProgress() {
        return this.run_status.status === 'in progress'
    }
}

export enum RunStatuses {
    running = 'running'
}