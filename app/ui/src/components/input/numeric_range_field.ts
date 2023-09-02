import {WeyaElementFunction} from "../../../../lib/weya/weya"
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
                            autocomplete: this.autocomplete,
                        }
                    )
                this.inputElem.required = this.required
                this.inputElem.disabled = this._disabled
            })
        })
    }
}

interface NumericRangeFieldOptions {
    min: number
    max: number
    onClick: (min: number, max: number) => void
    buttonLabel: string
}

export class NumericRangeField {

    private readonly minField: NumericEditableField
    private readonly maxField: NumericEditableField
    private readonly onClick: (min: number, max: number) => void
    private readonly doneButton: CustomButton

    constructor(opt: NumericRangeFieldOptions) {
        this.onClick = opt.onClick
        this.minField = new NumericEditableField({name: "", value: `${opt.min}`, placeholder: "From"})
        this.maxField = new NumericEditableField({name: "", value: `${opt.max}`, placeholder: "To"})
        this.doneButton = new CustomButton({
            onButtonClick: () => {
                let range: number[] = this.getRange()
                this.onClick(range[0], range[1])
            }, parent: this.constructor.name, text: opt.buttonLabel, noMargin: true
        })
    }

    public getRange(): [number, number] {
        return [parseFloat(this.minField.getInput()), parseFloat(this.maxField.getInput())]
    }

    public setRange(min: number, max: number) {
        this.minField.updateValue(`${min}`)
        this.maxField.updateValue(`${max}`)
    }

    render($: WeyaElementFunction) {
        $('div.range-field', $=> {
            this.minField.render($)
            this.maxField.render($)
            this.doneButton.render($)
        })
    }
}