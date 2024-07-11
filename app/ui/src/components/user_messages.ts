import {WeyaElementFunction, Weya as $, WeyaElement,} from '../../../lib/weya/weya'
import {NetworkError} from "../network"
import {errorToString} from "../utils/value"

export class UserMessages {
    message: string
    elem: WeyaElement

    public static shared = new UserMessages()

    private constructor() {
    }

    private render() {
        if (document.getElementById("shared-u-m") != null)
            return
        this.elem = $('div#shared-u-m', '.pointer-cursor.mt-1')
        document.body.prepend(this.elem)
    }

    hide(isHidden: boolean) {
        this.render()
        if (isHidden) {
            this.elem.classList.add('hide')
        } else {
            this.elem.classList.remove('hide')
        }
    }

    error(message: string = 'An unexpected network error occurred. Please try again later') {
        this.render()
        this.message = message
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', '.message.alert', $ => {
                $('span', this.message.substring(0, 100))
                if (this.message.length > 100) {
                    $('a', '',{
                        href: '#',
                        on: {
                            click: () => {
                                let win = window.open('', '_blank')
                                win.document.body.innerHTML = `<pre>${this.message}</pre>`
                            }
                        }
                    }, $ => {
                        $('span', ' ...more')
                    })
                }

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
            description += errorToString(error)
        }
        this.error(description)
    }

    success(message: string) {
        this.render()
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
        this.render()
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
