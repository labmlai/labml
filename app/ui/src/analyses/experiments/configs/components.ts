import {FormattedValue} from "../../../utils/value"
import {Config} from "../../../models/config"
import {WeyaElement, WeyaElementFunction} from "../../../../../lib/weya/weya"


export const CONFIG_PRINT_LEN = 20
export const KEY_WIDTH = 125
export const PADDING = 11

export class ComputedValue {
    computed: any

    constructor(opt: { computed: any }) {
        this.computed = opt.computed
    }

    render($: WeyaElementFunction) {
        if (typeof this.computed !== 'string') {
            $('span.computed', $ => {
                new FormattedValue({value: this.computed}).render($)
            })
            return
        }

        this.computed = this.computed.replace('\n', '')
        if (this.computed.length < CONFIG_PRINT_LEN) {
            $('span.computed', this.computed)
            return
        }

        let truncated = this.computed.substr(0, CONFIG_PRINT_LEN) + '...'
        let split = this.computed.split('.')
        if (this.computed.indexOf(' ') === -1 && split.length > 1) {
            truncated = '...' + split[split.length - 1]
            if (truncated.length > CONFIG_PRINT_LEN) {
                truncated = truncated.substr(0, CONFIG_PRINT_LEN) + '...'
            }
        }
        $('span.computed', $ => {
            $('span.empty', truncated, {title: this.computed})
        })
    }
}

class Option {
    value: any

    constructor(opt: { value: any }) {
        this.value = opt.value
    }

    render($: WeyaElementFunction) {
        $('span.option', this.value)
    }
}

class OtherOptions {
    options: any[]

    constructor(opt: { options: any[] }) {
        this.options = opt.options
    }

    render($: WeyaElementFunction) {
        if (this.options.length === 0) {
            return
        }

        $('span.options', $ => {
            this.options.filter((o => typeof o === 'string')).map((o) =>
                $('span', o)
            )
        })
    }
}

interface ConfigItemOptions {
    config: Config
    configs: Config[]
    isHyperParamOnly: boolean
    width: number
}

class ConfigItemView {
    conf: Config
    isHyperParamOnly: boolean
    isParentDefault: boolean
    classes: string[]
    key: string
    width: number
    elem: WeyaElement

    constructor(opt: ConfigItemOptions) {
        this.conf = opt.config
        let configs: { [key: string]: Config } = {}
        for (let c of opt.configs) {
            configs[c.key] = c
        }
        this.width = opt.width
        this.isHyperParamOnly = opt.isHyperParamOnly

        this.classes = ['info_list', 'config']

        let prefix = ''
        let parentKey = ''
        this.isParentDefault = false
        let conf_modules = this.conf.key.split('.')
        for (let i = 0; i < conf_modules.length - 1; ++i) {
            parentKey += conf_modules[i]
            if (configs[parentKey] && configs[parentKey].isDefault) {
                this.isParentDefault = true
            }
            parentKey += '.'
            prefix += '--- '
        }

        this.key = prefix + this.conf.name
        if (opt.isHyperParamOnly) {
            this.key = this.conf.key
        }
    }

    render($: WeyaElementFunction) {
        if (this.conf.order < 0) {
            this.classes.push('ignored')
            if (this.isHyperParamOnly) {
                return
            }
        }

        if (this.conf.isMeta) {
            return
        }

        if (!this.conf.isExplicitlySpecified && !this.conf.isHyperparam) {
            if (this.isHyperParamOnly) {
                return
            }
        }

        this.elem = $('div', $ => {
            $('span.key', this.key)
            $('span.combined', {style: {width: `${this.width - KEY_WIDTH - 2 * PADDING}px`}}, $ => {
                new ComputedValue({computed: this.conf.computed}).render($)

                if (this.conf.isCustom) {
                    if (this.isParentDefault) {
                        this.classes.push('only_option')
                    } else {
                        this.classes.push('custom')
                    }
                } else {
                    new Option({value: this.conf.value}).render($)

                    if (this.isParentDefault || this.conf.isOnlyOption) {
                        this.classes.push('only_option')
                    } else {
                        this.classes.push('picked')
                    }
                }

                if (!this.isHyperParamOnly && this.conf.otherOptions) {
                    new OtherOptions({options: [...this.conf.otherOptions]}).render($)
                }

                if (this.conf.isHyperparam) {
                    this.classes.push('hyperparam')
                } else if (this.conf.isExplicitlySpecified) {
                    this.classes.push('specified')
                } else {
                    this.classes.push('not-hyperparam')
                }
            })
        })

        for (let cls of this.classes) {
            this.elem.classList.add(cls)
        }
    }
}

interface ConfigsOptions {
    configs: Config[]
    isHyperParamOnly: boolean
    width: number
}

export class Configs {
    configs: Config[]
    isHyperParamOnly: boolean
    width: number
    count: number

    constructor(opt: ConfigsOptions) {
        this.configs = opt.configs
        this.isHyperParamOnly = opt.isHyperParamOnly
        this.width = opt.width

        this.configs.sort((a, b) => {
            if (a.key < b.key) return -1;
            else if (a.key > b.key) return +1;
            else return 0
        })

        this.count = this.configs.length
        if (opt.isHyperParamOnly) {
            this.count = this.configs.filter((c) => {
                return !(c.order < 0 ||
                    (!c.isExplicitlySpecified && !c.isHyperparam))
            }).length
        }
    }

    render($: WeyaElementFunction) {
        $('div','.configs.block.collapsed', {style: {width: `${this.width}px`}}, $ => {
            if (this.count === 0 && this.isHyperParamOnly) {
                $('div','.info', 'Default configurations')
                return
            }

            this.configs.map((c) =>
                new ConfigItemView({
                    config: c,
                    configs: this.configs,
                    width: this.width,
                    isHyperParamOnly: this.isHyperParamOnly
                }).render($)
            )
        })
    }
}
