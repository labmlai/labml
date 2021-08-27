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
        githubIssues: () => {
            window.open('https://github.com/labmlai/labml/issues')
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
                    {on: {click: this.events.githubIssues}},
                    $ => {
                        $('span', '.far.fa-dot-circle', '')
                        $('span', '.m-1', 'Reach us on Github issues')
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
