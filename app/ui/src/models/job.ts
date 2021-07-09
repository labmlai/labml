export interface JobModel {
    job_uuid: string
    method: string
    status: string
    created_time: number
    completed_time: number
    data: object
}


export class Job {
    job_uuid: string
    method: string
    status: string
    created_time: number
    completed_time: number
    data: object

    constructor(job: JobModel) {
        this.job_uuid = job.job_uuid
        this.method = job.method
        this.status = job.status
        this.created_time = job.created_time
        this.completed_time = job.completed_time
        this.data = job.data
    }

    get isSuccessful() {
        return this.status === 'success'
    }

    get isFailed() {
        return this.status === 'fail'
    }

    get isTimeOut() {
        return this.status === 'timeout'
    }

    get isComputerOffline() {
        return this.status === 'computer_offline'
    }
}