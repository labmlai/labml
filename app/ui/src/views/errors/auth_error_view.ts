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

class AuthErrorView extends ScreenView {
    elem: HTMLDivElement
    private events = {
        back: () => {
            if (ROUTER.canBack()) {
                ROUTER.back()
            } else {
                ROUTER.navigate('/')
            }
        },
        login: () => {
            ROUTER.navigate(`/login`)
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

        mix_panel.track('401 View')
    }

    get requiresAuth(): boolean {
        return false
    }

    render() {
        setTitle({section: '401'})
        this.elem = $('div', '.error-container', $ => {
            $('h2', '.mt-5', 'Ooops! Authentication Failure.' + '')
            $('h1', '401')
            $('p', 'We are having trouble authenticating your request' + '')
            $('div', '.btn-container.mt-3', $ => {
                $('button', '.btn.nav-link',
                    {on: {click: this.events.back}},
                    $ => {
                        $('span', '.fas.fa-redo', '')
                        $('span', '.m-1', 'Retry')
                    })
                $('button', '.btn.nav-link',
                    {on: {click: this.events.login}},
                    $ => {
                        $('span', '.fas.fa-user', '')
                        $('span', '.m-1', 'Login Again')
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

export class AuthErrorHandler {
    constructor() {
        ROUTER.route('401', [AuthErrorHandler.handleAuthError])
    }

    static handleAuthError = () => {
        SCREEN.setView(new AuthErrorView())
    }
}
