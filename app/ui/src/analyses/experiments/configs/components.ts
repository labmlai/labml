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
    isSummary: boolean
    width: number
    onTap?: (key: string) => void
}

export class ConfigItemView {
    conf: Config
    isParentDefault: boolean
    classes: string[]
    key: string
    width: number
    elem: WeyaElement
    onTap?: (key: string) => void
    isSummary: boolean

    constructor(opt: ConfigItemOptions) {
        this.conf = opt.config
        let configs: { [key: string]: Config } = {}
        for (let c of opt.configs) {
            configs[c.key] = c
        }
        this.width = opt.width
        this.onTap = opt.onTap

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
        this.isSummary = opt.isSummary
        if (opt.isSummary) {
            this.key = this.conf.key
        }
    }

    render($: WeyaElementFunction) {
        if (this.conf.order < 0) {
            this.classes.push('ignored')
        }

        if (this.conf.isMeta) {
            return
        }

        this.elem = $('div', {on: {click: () => {
                    if (this.onTap == null) {
                        return
                    }
                    this.conf.isSelected = !this.conf.isSelected
                    if (this.conf.isSelected) {
                        this.elem.classList.add('selected')
                    } else {
                        this.elem.classList.remove('selected')
                    }
                    this.onTap(this.conf.key)
                }}}, $ => {
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

                if (!this.isSummary && this.conf.otherOptions) {
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

        if (this.conf.isSelected && !this.isSummary) {
            this.elem.classList.add('selected')
        }
    }
}

interface ConfigsOptions {
    configs: Config[]
    isSummary: boolean
    width: number
    onTap?: (key: string) => void
}

export class Configs {
    configs: Config[]
    isSummary: boolean
    width: number
    onTap?: (key: string)=>void

    constructor(opt: ConfigsOptions) {
        this.configs = opt.configs
        this.isSummary = opt.isSummary
        this.width = opt.width
        this.onTap = opt.onTap

        this.configs.sort((a, b) => {
            if (a.key < b.key) return -1;
            else if (a.key > b.key) return +1;
            else return 0
        })

        if (opt.isSummary) {
            this.configs = this.configs.filter((c) => {  // show selected configs
                return c.isSelected
            })

            if (this.configs.length == 0) {  // show hyper params only as default
                if (opt.isSummary) {
                    this.configs = this.configs.filter((c) => {
                        return !(c.order < 0 ||
                            (!c.isExplicitlySpecified && !c.isHyperparam))
                    })
                }
            }
        }
    }

    render($: WeyaElementFunction) {
        $('div','.configs.block.collapsed', {style: {width: `${this.width}px`}}, $ => {
            if (this.configs.length === 0 && this.isSummary) {
                $('div','.info', 'Default configurations')
                return
            }

            this.configs.map((c) =>
                new ConfigItemView({
                    config: c,
                    configs: this.configs,
                    width: this.width,
                    onTap: this.onTap,
                    isSummary: this.isSummary
                }).render($)
            )
        })
    }
}
