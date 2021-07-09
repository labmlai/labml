import {WeyaElementFunction, Weya as $,} from '../../../lib/weya/weya'

export class UserMessages {
    message: string
    elem: HTMLDivElement

    constructor() {
    }

    render($: WeyaElementFunction) {
        this.elem = $('div', '.pointer-cursor.mt-1')
    }

    hide(isHidden: boolean) {
        if (isHidden) {
            this.elem.classList.add('hide')
        } else {
            this.elem.classList.remove('hide')
        }
    }

    networkError() {
        this.message = 'An unexpected network error occurred. Please try again later'
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', '.message.alert', $ => {
                $('span', this.message)
                $('span', '.close-btn',
                    String.fromCharCode(215),
                    {on: {click: this.hide.bind(this, true)}}
                )
            })
        })
        this.hide(false)
    }

    success(message: string) {
        this.message = message
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', '.message.success', $ => {
                $('span', this.message)
                $('span', '.close-btn',
                    String.fromCharCode(215),
                    {on: {click: this.hide.bind(this, true)}}
                )
            })
        })
        this.hide(false)
    }

    warning(message: string) {
        this.message = message
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', '.message.alert', $ => {
                $('span', this.message)
                $('span', '.close-btn',
                    String.fromCharCode(215),
                    {on: {click: this.hide.bind(this, true)}}
                )
            })
        })
        this.hide(false)
    }
}
