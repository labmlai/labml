import {WeyaElementFunction} from '../../../lib/weya/weya'
import {RunStatusModel} from "../models/status"
import {ContentType} from '../types'

export interface StatusOptions {
    status: RunStatusModel
    type?: ContentType
    isDistributed?: boolean
}

export class StatusView {
    status: RunStatusModel
    type: ContentType
    isDistributed: boolean

    private readonly statusText: string
    private readonly statusClass: string

    constructor(opt: StatusOptions) {
        this.status = opt.status
        this.type = opt.type || 'run'
        this.isDistributed = opt.isDistributed || false

        if (this.status.status === 'in progress') {
            if (this.type === 'session') {
                this.statusClass = 'text-info'
                this.statusText = 'monitoring'
            } else {
                this.statusClass = 'text-info'
                this.statusText = 'experiment is running'
            }
        } else if (this.status.status === 'no response') {
            this.statusClass = 'text-warning'
            this.statusText = 'no response'
        } else if (this.status.status === 'completed') {
            this.statusClass = 'text-success'
            this.statusText = 'completed'
        } else if (this.status.status === 'crashed') {
            this.statusClass = 'text-danger'
            this.statusText = 'crashed'
        } else if (this.status.status === 'unknown') {
            this.statusClass = 'text-secondary'
            this.statusText = 'unknown status'
        } else if (this.status.status === 'interrupted'){
            this.statusClass = 'text-secondary'
            this.statusText = 'interrupted'
        } else {
            this.statusClass = 'text-secondary'
            this.statusText = 'unknown status'
        }
    }

    render($: WeyaElementFunction) {
        $('div.status', $=> {
            $(`div.text-uppercase.${this.statusClass}`, this.statusText)
            if (this.isDistributed) {
                $(`div.text-uppercase.label-info.label.text-light`, "distributed")
            }
        })
    }
}

