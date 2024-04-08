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
    private events = {
        retry: () => {
            if (ROUTER.canBack()) {
                ROUTER.back()
            }
        },
    }

    constructor() {
        super()
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
            $('p', 'There\'s a problem with the connection ' + '')
            $('div', '.btn-container.mt-3', $ => {
                $('button', '.btn.nav-link',
                    {on: {click: this.events.retry}},
                    $ => {
                        $('span', '.fas.fa-redo', '')
                        $('span', '.m-1', 'Retry')
                    })
            })
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

    static handleNetworkError = () => {
        SCREEN.setView(new NetworkErrorView())
    }
}
