import {WeyaElementFunction} from '../../../lib/weya/weya'
import {RunListItem} from '../models/run_list'
import {StatusView} from './status'
import {getDateTimeComponents} from '../utils/time'
import {ConfigItemView} from "../analyses/experiments/configs/components";
import {TagView} from "./tag"

export interface RunsListItemOptions {
    item: RunListItem
    onClick?: (elem: RunsListItemView) => void
    width: number
}

export class RunsListItemView {
    public elem: HTMLElement
    public item: RunListItem
    private readonly onClick: (evt: Event) => void
    private readonly width: number
    static readonly METRIC_LIMIT = 3

    constructor(opt: RunsListItemOptions) {
        this.item = opt.item
        this.width = opt.width
        if (opt.onClick) {
            this.onClick = (e: Event) => {
                e.preventDefault()
                opt.onClick(this)
            }
        }
    }

    private getTimeString() {
        if (this.item.start_time == null || this.item.last_updated_time == null) {
            return ''
        }

        let startDate = getDateTimeComponents(new Date(this.item.start_time * 1000))
        let endDate = getDateTimeComponents(new Date(this.item.last_updated_time * 1000))

        let timeString = `'${startDate[0]} ${startDate[1]} ${startDate[2]}, ${startDate[3]}:${startDate[4]}`

        if (startDate[0] == endDate[0] && startDate[1] == endDate[1] && startDate[2] == endDate[2]) {
            timeString += ` - ${endDate[3]}:${endDate[4]}`
        } else if (startDate[0] == endDate[0] && startDate[1] == endDate[1]) {
            timeString += ` - ${endDate[2]}, ${endDate[3]}:${endDate[4]}`
        } else if (startDate[0] == endDate[0]) {
            timeString += ` - ${endDate[1]} ${endDate[2]}, ${endDate[3]}:${endDate[4]}`
        } else {
            timeString += ` - ${endDate[0]} ${endDate[1]} ${endDate[2]}, ${endDate[3]}:${endDate[4]}`
        }

        return timeString
    }

    render($: WeyaElementFunction) {
        this.elem = $('a', '.list-item.list-group-item.list-group-item-action',
            {href: `/run/${this.item.run_uuid}`, on: {click: this.onClick}, target: '_blank'},
            $ => {
                if (this.item.run_status) {
                    new StatusView({status: this.item.run_status, isDistributed: this.item.world_size>0}).render($)
                }
                $('div', '.spaced-row', $ => {
                    $('div', $ => {
                        $('p.time', this.getTimeString())
                        $('div.tags', $ => {
                            this.item.tags.map((tag: any, _: any) => (
                                new TagView({text: tag}).render($)
                            ))
                        })
                        $('h5', this.item.name)
                        $('h6', this.item.comment)
                    })
                    $('div', $ => {
                        $('div.info_list.config.custom.label', $ => {
                            if (this.item.step != null) {
                                $('span',  `${this.item.step} Steps`)
                            }
                        })
                    })
                })
                $('div', '.spaced-row', $ => {

                        if (this.item.metric_values != null && this.item.metric_values.length != 0) {
                            $('div', $ => {
                                $('span', 'Metrics: ')
                                this.item.metric_values.slice(0, RunsListItemView.METRIC_LIMIT).map((m, idx) => {
                                    $('div.info_list.config.custom', $ => {
                                        $('span.key', m.name)
                                        $('span', `${m.value.toExponential(4)}`)
                                    })
                                })
                                if (this.item.metric_values.length > RunsListItemView.METRIC_LIMIT) {
                                    $('div.break.text-secondary', `+${this.item.metric_values.length - RunsListItemView.METRIC_LIMIT} more`)
                                }
                            })
                        }


                    $('div.configs', $ => {
                        if (this.item.favorite_configs != null && this.item.favorite_configs.length != 0) {
                            this.item.favorite_configs.map((c) => {
                                new ConfigItemView({
                                    config: c,
                                    configs: this.item.favorite_configs,
                                    width: this.width-20,
                                    onTap: undefined,
                                    isSummary: true
                                }).render($)
                            })
                        }
                    })
                })
            })
    }
}
