import {ROUTER, SCREEN} from '../../app'
import {Weya as $} from '../../../../lib/weya/weya'
import {setTitle} from '../../utils/document'
import {ScreenView} from '../../screen_view'

function wrapEvent(eventName: string, func: Function) {
    function wrapper() {
        let e: Event = arguments[arguments.length - 1]
        if (eventName[eventName.length - 1] !== '_') {
            e.preventDefault()
            e.stopPropagation()
        }

        func.apply(null, arguments)
    }

    return wrapper
}

class NetworkErrorView extends ScreenView {
    elem: HTMLDivElement
    error: Error

    private events = {
        retry: () => {
            if (ROUTER.canBack()) {
                ROUTER.back()
            }
        },
    }

    constructor(error: Error) {
        super()
        this.error = error
        let events = []
        for (let k in this.events) {
            events.push(k)
        }

        for (let k of events) {
            let func = this.events[k]
            this.events[k] = wrapEvent(k, func)
        }
    }

    get requiresAuth(): boolean {
        return false
    }

    render() {
        setTitle({section: 'Network Error'})
        this.elem = $('div', '.error-container', $ => {
            $('h2', '.mt-5', 'Ooops!' + '')
            $('p', 'we\'ve encountered an UI Error ' + '')
            if (this.error != null) {
                $('div.code-sample.bg-dark.px-1.py-2.my-3', $ => {
                    if (this.error.name != null) {
                    $('pre.text-white', this.error.name)
                    }
                    if (this.error.message != null) {
                        $('pre.text-white', this.error.message)
                    }
                    if (this.error.stack != null) {
                        $('pre.text-white', this.error.stack)
                    }
                })
            }
        })

        return this.elem
    }

    destroy() {
    }
}

export class NetworkErrorHandler {
    constructor() {
        ROUTER.route('network_error', [NetworkErrorHandler.handleNetworkError])
    }

    static handleNetworkError = (error: Error = null) => {
        SCREEN.setView(new NetworkErrorView(error))
    }
}
