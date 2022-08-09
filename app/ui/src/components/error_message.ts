import {Weya} from "../../../lib/weya/weya"

export class ErrorMessage {
    elem: HTMLDivElement
    private readonly text: string

    constructor(text?: string) {
        this.elem = null
        this.text = text ?? 'Network error'
    }

    render(parent: HTMLDivElement) {
        this.remove()
        Weya(parent, $ => {
            this.elem = $('div', '.error.text-center.warning', $ => {
                $('span', '.fas.fa-exclamation-triangle', '')
                $('h4', '.text-uppercase', this.text)
            })
        })
    }

    remove() {
        if (this.elem == null) {
            return
        }
        this.elem.remove()
        this.elem = null
    }
}
