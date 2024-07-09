import {WeyaElementFunction, Weya as $,} from '../../../lib/weya/weya'
import {NetworkError} from "../network"
import {errrToString} from "../utils/value"

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

    error(message: string = 'An unexpected network error occurred. Please try again later') {
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

    networkError(error: NetworkError | Error, message: String = 'An unexpected network error occurred. Please try again later') {
        let description = message + '\n'
        if (error instanceof NetworkError) {
            description += error.toString()
        } else {
            description += errrToString(error)
        }
        this.error(description)
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
