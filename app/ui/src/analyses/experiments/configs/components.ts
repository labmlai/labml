import {FormattedValue} from "../../../utils/value"
import {Config} from "../../../models/config"
import {WeyaElement, WeyaElementFunction} from "../../../../../lib/weya/weya"
import {ControlledToggleButton} from "../../../components/buttons";


export const CONFIG_PRINT_LEN = 20
export const KEY_WIDTH = 125
export const PADDING = 11

export enum ConfigStatus {
    SELECTED = 'selected',
    FAVOURITE = 'favourite',
    NONE = 'none'
}

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
        $('span.computed', $ => {
            $('span.empty', this.computed, {title: this.computed})
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
    onTap?: (key: string, configStatus: ConfigStatus) => void
}

export class ConfigItemView {
    conf: Config
    isParentDefault: boolean
    classes: string[]
    key: string
    width: number
    elem: WeyaElement
    onTap?: (key: string, configStatus: ConfigStatus) => void
    isSummary: boolean

    private selectToggle: ControlledToggleButton
    private favouriteToggle: ControlledToggleButton

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

        this.selectToggle = new ControlledToggleButton({
            icon: ".fas.fa-eye", parent: this.constructor.name, text: "",
            isToggled: this.conf.isSelected,
            onButtonClick: () => {
                this.onTapHandler(ConfigStatus.SELECTED)
            }
        })
        this.favouriteToggle = new ControlledToggleButton({
            icon: ".fas.fa-star", parent: this.constructor.name, text: "",
            isToggled: this.conf.isFavourite,
            onButtonClick: () => {
                this.onTapHandler(ConfigStatus.FAVOURITE)
            }
        })
    }

    private onTapHandler(configStatus: ConfigStatus): void {
        if (this.onTap == null) {
            return
        }
        if (configStatus === ConfigStatus.FAVOURITE) {
            if (this.conf.isFavourite) {
                this.conf.isFavourite = false
            } else {
                this.conf.isFavourite = true
                this.conf.isSelected = true
            }
        } else if (configStatus === ConfigStatus.SELECTED) {
            if (this.conf.isSelected) {
                this.conf.isSelected = false
                this.conf.isFavourite = false
            } else {
                this.conf.isSelected = true
            }
        }

        this.updateButtons()

        let newConfigStatus: ConfigStatus = ConfigStatus.NONE
        if (this.conf.isSelected) {
            newConfigStatus = ConfigStatus.SELECTED
        }
        if (this.conf.isFavourite) {
            newConfigStatus = ConfigStatus.FAVOURITE
        }
        this.onTap(this.conf.key, newConfigStatus)
    }

    private updateButtons() {
        this.selectToggle.toggle = this.conf.isSelected
        this.favouriteToggle.toggle = this.conf.isFavourite
    }

    render($: WeyaElementFunction) {
        if (this.conf.order < 0) {
            this.classes.push('ignored')
        }

        if (this.conf.isMeta) {
            return
        }

        this.elem = $('div',  $ => {
            if (!this.isSummary) {
                this.selectToggle.render($)
                this.favouriteToggle.render($)
            }
            $('span.key', this.key)
            $('span.combined', $ => {
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

        this.updateButtons()
    }
}

interface ConfigsOptions {
    configs: Config[]
    isSummary: boolean
    width: number
    onTap?: (key: string, configStatus: ConfigStatus) => void
}

export class Configs {
    configs: Config[]
    isSummary: boolean
    width: number
    onTap?: (key: string, configStatus: ConfigStatus)=>void

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
