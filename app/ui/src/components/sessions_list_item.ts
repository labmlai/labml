import {WeyaElementFunction} from '../../../lib/weya/weya'
import {StatusView} from './status'
import {formatTime} from '../utils/time'
import {SessionsListItemModel} from '../models/session_list';

export interface SessionListItemOptions {
    item: SessionsListItemModel
    onClick: (elem: SessionsListItemView) => void
}

export class SessionsListItemView {
    item: SessionsListItemModel
    elem: HTMLAnchorElement
    onClick: (evt: Event) => void

    constructor(opt: SessionListItemOptions) {
        this.item = opt.item
        this.onClick = (e: Event) => {
            e.preventDefault()
            opt.onClick(this)
        }
    }

    render($: WeyaElementFunction) {
        this.elem = $('a', '.list-item.list-group-item.list-group-item-action',
            {href: `/session/${this.item.session_uuid}`, on: {click: this.onClick}},
            $ => {
                $('div', $ => {
                    new StatusView({status: this.item.run_status, type: 'session'}).render($)
                    $('p', `Started ${formatTime(this.item.start_time)}`)
                    $('h5', this.item.name)
                    $('h6', this.item.comment)
                })
            })
    }
}
