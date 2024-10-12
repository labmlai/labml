import {WeyaElementFunction} from '../../../../lib/weya/weya'

export interface EditableFieldOptions {
    name: string
    value: any
    placeholder?: string
    isEditable?: boolean
    numEditRows?: number
    type?: string
    autocomplete?: string
    required?: boolean
    onChange?: (value: string) => void
}

export default class EditableField {
    name: string
    value: any
    placeholder: string
    isEditable: boolean
    numEditRows: number
    inputElem: HTMLInputElement | HTMLTextAreaElement
    valueElem: HTMLSpanElement
    protected type: string
    protected readonly autocomplete?: string
    protected readonly required: boolean
    protected readonly onChange?: (value: string) => void

    constructor(opt: EditableFieldOptions) {
        this.name = opt.name
        this.value = opt.value
        this.placeholder = opt.placeholder
        this.isEditable = opt.isEditable
        this.numEditRows = opt.numEditRows
        this.type = opt.type
        this.autocomplete = opt.autocomplete
        this.required = opt.required ?? false
        this.onChange = opt.onChange
    }

    protected _disabled: boolean

    set disabled(value: boolean) {
        this._disabled = value
        this.inputElem.disabled = value
    }

    getInput() {
        return this.inputElem.value
    }

    updateValue(value: string) {
        this.valueElem.textContent = value
    }

    updateInput(value: string) {
        this.inputElem.value = value
    }

    render($: WeyaElementFunction) {
        $(`li`, $ => {
            $('span.item-key', this.name)
            if (this.isEditable) {
                $('div.input-container.mt-2', $ => {
                    $('div.input-content', $ => {
                        if (this.numEditRows) {
                            this.inputElem = <HTMLTextAreaElement>$('textarea', {
                                    rows: this.numEditRows,
                                    placeholder: this.placeholder,
                                    autocomplete: this.autocomplete,
                                    type: this.type
                                }
                            )
                            this.inputElem.innerHTML = this.value.split('\n').join('&#013;')
                            this.inputElem.addEventListener('input', (event: KeyboardEvent) => {
                                if (this.onChange) {
                                    this.onChange(this.inputElem.value)
                                }
                            })
                        } else {
                            this.inputElem = <HTMLInputElement>$('input', {
                                    placeholder: this.placeholder,
                                    value: this.value,
                                    type: this.type,
                                    autocomplete: this.autocomplete,
                                }
                            )
                            this.inputElem.addEventListener('input', (event: KeyboardEvent) => {
                                if (this.onChange) {
                                    this.onChange(this.inputElem.value)
                                }
                            })
                        }
                        this.inputElem.required = this.required
                        this.inputElem.disabled = this._disabled
                    })
                })
            } else {
                this.valueElem = $('span', '.item-value')
                this.updateValue(this.value)
            }
        })
    }
}
