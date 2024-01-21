import {Weya} from "../../../lib/weya/weya"

export class ErrorMessage {
    elem: HTMLDivElement
    private readonly text: string
    private readonly error: string

    constructor(text?: string, error?: string) {
        this.elem = null
        this.text = text ?? 'Network error'
        this.error = error ?? ''
    }

    render(parent: HTMLDivElement) {
        this.remove()
        Weya(parent, $ => {
            this.elem = $('div', '.error.text-center.warning', $ => {
                $('span', '.fas.fa-exclamation-triangle', '')
                $('h4', '.text-uppercase', this.text)
                if (this.error != "") {
                    $('pre.text-secondary', this.error)
                }
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
