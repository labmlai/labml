import {WeyaElement, WeyaElementFunction} from "../../../../lib/weya/weya"
import EditableField, {EditableFieldOptions} from "./editable_field"
import {CustomButton} from "../buttons"

class NumericEditableField extends EditableField {
    constructor(opt: EditableFieldOptions) {
        super(opt)
        this.type = 'number'
        this.isEditable = true
    }

    render($: WeyaElementFunction) {
        $('div.input-container', $ => {
            $('div.input-content', $ => {
                this.inputElem = <HTMLInputElement>$('input', {
                            placeholder: this.placeholder,
                            value: this.value,
                            type: this.type,
                            autocomplete: this.autocomplete
                        }
                    )
                this.inputElem.required = this.required
                this.inputElem.disabled = this._disabled
                this.inputElem.oninput = () => {
                    if (this.onChange) {
                        this.onChange(this.inputElem.value)
                    }
                }
                this.inputElem.onchange = () => {
                    if (this.onChange) {
                        this.onChange(this.inputElem.value)
                    }
                }
            })
        })
    }
}

interface NumericRangeFieldOptions {
    min: number
    max: number
    onChange: () => void
    buttonLabel: string
}

export class NumericRangeField {

    private readonly minField: NumericEditableField
    private readonly maxField: NumericEditableField
    private readonly onChange: () => void
    private elem: WeyaElement
    private min: number
    private max: number

    constructor(opt: NumericRangeFieldOptions) {
        this.onChange = opt.onChange
        this.min = opt.min
        this.max = opt.max
        this.minField = new NumericEditableField({
            name: "",
            value: `${this.min}`,
            placeholder: "From", onChange: () => {
                this.onChange()
            }})
        this.maxField = new NumericEditableField({
            name: "",
            value: `${this.max}`,
            placeholder: "To", onChange: () => {
                this.onChange()
            }})
    }

    public getRange(): [number, number] {
        let min = this.minField.getInput() == '' ? -1 : parseFloat(this.minField.getInput())
        let max = this.maxField.getInput() == '' ? -1 : parseFloat(this.maxField.getInput())

        this.min = min
        this.max = max

        return [min, max]
    }

    public setRange(min: number, max: number) {
        this.min = min
        this.max = max

        if (this.elem == null || !this.minField.valueElem) { //element is not yet rendered or destroyed
            return
        }

        this.minField.updateValue(min == -1 ? '' : `${min}`)
        this.maxField.updateValue(max == -1 ? '' : `${max}`)
    }

    render($: WeyaElementFunction) {
        this.minField.value = this.min == -1 ? '' : `${this.min}`
        this.maxField.value = this.max == -1 ? '' : `${this.max}`

        this.elem = $('div.range-field', $=> {
            this.minField.render($)
            this.maxField.render($)
        })
    }
}