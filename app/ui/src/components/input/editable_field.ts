import {WeyaElementFunction} from '../../../../lib/weya/weya'

interface EditableFieldOptions {
    name: string
    value: any
    placeholder?: string
    isEditable?: boolean
    numEditRows?: number
}

export default class EditableField {
    name: string
    value: any
    placeholder: string
    isEditable: boolean
    numEditRows: number
    inputElem: HTMLInputElement | HTMLTextAreaElement
    valueElem: HTMLSpanElement

    constructor(opt: EditableFieldOptions) {
        this.name = opt.name
        this.value = opt.value
        this.placeholder = opt.placeholder
        this.isEditable = opt.isEditable
        this.numEditRows = opt.numEditRows
    }

    getInput() {
        return this.inputElem.value
    }

    updateValue(value: string) {
        this.valueElem.textContent = value
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
                                    value: this.value
                                }
                            )
                        } else {
                            this.inputElem = <HTMLInputElement>$('input', {
                                    placeholder: this.placeholder,
                                    value: this.value
                                }
                            )
                        }
                    })
                })
            } else {
                this.valueElem = $('span', '.item-value')
                this.updateValue(this.value)
            }
        })
    }
}
