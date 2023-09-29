import {WeyaElementFunction} from '../../../lib/weya/weya'
import {RunListItemModel} from '../models/run_list'
import {StatusView} from './status'
import {formatTime} from '../utils/time'
import {StandaloneSparkLine} from "./charts/spark_lines/spark_line";
import {getExtent, toPointValue} from "./charts/utils";

export interface RunsListItemOptions {
    item: RunListItemModel
    onClick: (elem: RunsListItemView) => void
}

export class RunsListItemView {
    public elem: HTMLElement
    public item: RunListItemModel
    private readonly onClick: (evt: Event) => void

    constructor(opt: RunsListItemOptions) {
        this.item = opt.item
        this.onClick = (e: Event) => {
            e.preventDefault()
            opt.onClick(this)
        }
    }

    private renderSparkLine($: WeyaElementFunction) {
        if (this.item.preview_series == null || this.item.preview_series.value == null) {
            return
        }
        $('div', $ => {
            new StandaloneSparkLine({
                name: "",
                series: toPointValue(this.item.preview_series),
                width: 300,
                stepExtent: getExtent([toPointValue(this.item.preview_series)], d => d.step)
            }).render($)
        })
    }

    render($: WeyaElementFunction) {
        this.elem = $('a', '.list-item.list-group-item.list-group-item-action.spaced-row',
            {href: `/run/${this.item.run_uuid}`, on: {click: this.onClick}},
            $ => {
                $('div', $ => {
                    new StatusView({status: this.item.run_status, isDistributed: this.item.world_size>0}).render($)
                    $('p', `Started on ${formatTime(this.item.start_time)}`)
                    $('h5', this.item.name)
                    $('h6', this.item.comment)
                })
                $('div', '.preview-series', $ => {
                    this.renderSparkLine($)
                })
            })
    }
}
