import {Config} from "../../../models/config"
import {WeyaElementFunction} from "../../../../../lib/weya/weya"
import {ComputedValue, KEY_WIDTH, PADDING} from "../../experiments/configs/components"


interface ConfigsOptions {
    configs: Config[]
    width: number
}

export class Configs {
    configs: Config[]
    width: number
    count: number

    constructor(opt: ConfigsOptions) {
        this.configs = opt.configs
        this.width = opt.width
    }

    render($: WeyaElementFunction) {
        $('div', '.configs.block.collapsed', {style: {width: `${this.width}px`}}, $ => {
            this.configs.map((c) =>
                $('div', '.info_list.config', $ => {
                    $('span.key', c.key)
                    $('span.combined', {style: {width: `${this.width - KEY_WIDTH - 2 * PADDING}px`}}, $ => {
                        new ComputedValue({computed: c.value}).render($)
                    })
                })
            )
        })
    }
}
