import {WeyaElementFunction} from '../../../lib/weya/weya'
import {RunListItemModel} from '../models/run_list'
import {StatusView} from './status'
import {formatTime} from '../utils/time'

export interface RunsListItemOptions {
    item: RunListItemModel
    onClick: (elem: RunsListItemView) => void
}

export class RunsListItemView {
    item: RunListItemModel
    elem: HTMLAnchorElement
    onClick: (evt: Event) => void

    constructor(opt: RunsListItemOptions) {
        this.item = opt.item
        this.onClick = (e: Event) => {
            e.preventDefault()
            opt.onClick(this)
        }
    }

    render($: WeyaElementFunction) {
        this.elem = $('a', '.list-item.list-group-item.list-group-item-action',
            {href: `/run/${this.item.run_uuid}`, on: {click: this.onClick}},
            $ => {
                $('div', $ => {
                    new StatusView({status: this.item.run_status, isDistributed: this.item.world_size>0}).render($)
                    $('p', `Started on ${formatTime(this.item.start_time)}`)
                    $('h5', this.item.name)
                    $('h6', this.item.comment)
                })
            })
    }
}
