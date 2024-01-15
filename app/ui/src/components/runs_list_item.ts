import {WeyaElementFunction} from '../../../lib/weya/weya'
import {RunListItem} from '../models/run_list'
import {StatusView} from './status'
import {formatTime} from '../utils/time'
import {StandaloneSparkLine} from "./charts/spark_lines/spark_line";
import {getExtent, toPointValue} from "./charts/utils";
import {ConfigItemView} from "../analyses/experiments/configs/components";

export interface RunsListItemOptions {
    item: RunListItem
    onClick: (elem: RunsListItemView) => void
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
        this.onClick = (e: Event) => {
            e.preventDefault()
            opt.onClick(this)
        }
    }

    render($: WeyaElementFunction) {
        this.elem = $('a', '.list-item.list-group-item.list-group-item-action',
            {href: `/run/${this.item.run_uuid}`, on: {click: this.onClick}},
            $ => {
                new StatusView({status: this.item.run_status, isDistributed: this.item.world_size>0}).render($)
                $('div', '.spaced-row', $ => {
                    $('div', $ => {
                        $('p', `Started on ${formatTime(this.item.start_time)}`)
                        $('h5', this.item.name)
                        $('h6', this.item.comment)
                    })
                })
                if (this.item.favorite_configs != null) {
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
    }
}
