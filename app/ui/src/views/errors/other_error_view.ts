import {ROUTER, SCREEN} from '../../app'
import {Weya as $} from '../../../../lib/weya/weya'
import mix_panel from "../../mix_panel"
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

class OtherErrorView extends ScreenView {
    elem: HTMLDivElement
    private events = {
        back: () => {
            if (ROUTER.canBack()) {
                ROUTER.back()
            } else {
                ROUTER.navigate('/')
            }
        },
        slack: () => {
            window.open('https://join.slack.com/t/labforml/shared_invite/zt-egj9zvq9-Dl3hhZqobexgT7aVKnD14g/')
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

        mix_panel.track('500 View')
    }

    get requiresAuth(): boolean {
        return false
    }

    render() {
        setTitle({section: '500'})
        this.elem = $('div', '.error-container', $ => {
            $('h2', '.mt-5', 'Ooops! Something went wrong' + '')
            $('h1', '500')
            $('p', 'Seems like we are having issues right now' + '')
            $('div', '.btn-container.mt-3', $ => {
                $('button', '.btn.nav-link',
                    {on: {click: this.events.back}},
                    $ => {
                        $('span', '.fas.fa-redo', '')
                        $('span', '.m-1', 'Retry')
                    })
                $('button', '.btn.nav-link',
                    {on: {click: this.events.slack}},
                    $ => {
                        $('span', '.fas.fa-comments', '')
                        $('span', '.m-1', 'Reach us on Slack')
                    })
            })

        })

        return this.elem
    }

    destroy() {
    }
}

export class OtherErrorHandler {
    constructor() {
        ROUTER.route('500', [OtherErrorHandler.handleOtherError])
    }

    static handleOtherError = () => {
        SCREEN.setView(new OtherErrorView())
    }
}
