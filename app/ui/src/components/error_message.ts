import {Weya} from "../../../lib/weya/weya"

export class ErrorMessage {
    elem: HTMLDivElement

    constructor() {
        this.elem = null
    }

    render(parent: HTMLDivElement) {
        this.remove()
        Weya(parent, $ => {
            this.elem = $('div', '.error.text-center.warning', $ => {
                $('span', '.fas.fa-exclamation-triangle', '')
                $('h4', '.text-uppercase', 'Network error')
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
