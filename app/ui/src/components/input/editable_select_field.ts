import {WeyaElementFunction} from '../../../../lib/weya/weya'

interface EditableSelectFieldOptions {
    name: string
    value: any
    isEditable?: boolean
    onEdit: () => void
    onClick?: () => void
}

export default class EditableSelectField {
    name: string
    value: any
    isEditable: boolean
    private readonly onEdit: () => void
    private readonly _onClick?: () => void

    constructor(opt: EditableSelectFieldOptions) {
        this.name = opt.name
        this.value = opt.value
        this.isEditable = opt.isEditable
        this.onEdit = opt.onEdit
        this._onClick = opt.onClick
    }

    onClick() {
        if (!this.isEditable) {
            this._onClick()
        }
    }

    render($: WeyaElementFunction) {
        $(`li`, {on: {click: this.onClick.bind(this)}}, $ => {
            $('span.item-key', this.name)
            if (this.isEditable) {
                $('div.input-container.mt-2', $ => {
                    $('div', '.input-content', {on: {click: this.onEdit}}, $ => {
                        $('input', {
                                value: this.value,
                                readonly: 'readonly'
                            }
                        )
                    })
                })
            } else {
                $('span.item-value', this.value)
            }
        })
    }
}
