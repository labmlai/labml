export interface ConfigModel {
    key: string
    name: string
    computed: any
    value: any
    options: string[]
    order: number
    type: string
    is_hyperparam?: boolean
    is_meta?: boolean
    is_explicitly_specified?: boolean
}

export class Config {
    key: string
    name: string
    computed: any
    value: any
    options: string[]
    order: number
    type: string
    isHyperparam?: boolean
    isMeta?: boolean
    isExplicitlySpecified?: boolean

    isCustom: boolean
    isOnlyOption: boolean
    isDefault: boolean
    otherOptions: Set<string>

    isSelected: boolean
    isFavourite: boolean

    constructor(config: ConfigModel, isSelected?: boolean, isFavourite?: boolean) {
        this.key = config.key
        this.name = config.name
        this.computed = config.computed
        this.value = config.value
        this.options = config.options ? config.options: []
        this.order = config.order
        this.type = config.type
        this.isHyperparam = config.is_hyperparam
        this.isMeta = config.is_meta
        this.isExplicitlySpecified = config.is_explicitly_specified
        this.isFavourite = isFavourite ?? false
        this.isSelected = isSelected ?? false

        let options = new Set<string>()
        for (let opt of this.options) {
            options.add(opt)
        }

        this.isCustom = false
        this.isOnlyOption = false
        this.isDefault = false

        if (options.has(this.value)) {
            options.delete(this.value)
            if (options.size === 0) {
                this.isOnlyOption = true
                this.isDefault = true
            }
        } else {
            this.isCustom = true
            if (this.isExplicitlySpecified !== true) {
                this.isDefault = true
            }
        }

        this.otherOptions = options
    }
}
