import {WeyaElementFunction} from '../../../lib/weya/weya'
import {RunStatusModel} from "../models/status"
import {ContentType} from '../types';

export interface StatusOptions {
    status: RunStatusModel
    type?: ContentType
}

export class StatusView {
    status: RunStatusModel
    type: ContentType

    constructor(opt: StatusOptions) {
        this.status = opt.status
        this.type = opt.type || 'run'
    }

    render($: WeyaElementFunction) {
        if (this.status.status === 'in progress') {
            if (this.type === 'session') {
                $('div.status.text-info.text-uppercase', 'monitoring')
                return
            }
            $('div.status.text-info.text-uppercase', 'experiment is running')
        } else if (this.status.status === 'no response') {
            $('div.status.text-warning.text-uppercase', 'no response')
        } else if (this.status.status === 'completed') {
            $('div.status.text-success.text-uppercase', 'completed')
        } else if (this.status.status === 'crashed') {
            $('div.status.text-danger.text-uppercase', 'crashed')
        } else if (this.status.status === 'unknown') {
            $('div.status.text-info.text-uppercase', 'unknown status')
        } else {
            $('div.status', this.status.status)
        }
    }
}

