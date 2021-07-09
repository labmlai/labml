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

class PageNotFoundView extends ScreenView {
    elem: HTMLDivElement
    private events = {
        home: () => {
            ROUTER.navigate(`/`)
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

        mix_panel.track('404 View')
    }

    get requiresAuth(): boolean {
        return false
    }

    render() {
        setTitle({section: '404'})
        this.elem = $('div', '.error-container', $ => {
            $('h2', '.mt-5', 'Ooops! Page not found.' + '')
            $('h1', '404')
            $('p', 'We can\'t find the page.' + '')
            $('div', '.btn-container.mt-3', $ => {
                $('button', '.btn.nav-link',
                    {on: {click: this.events.home}},
                    $ => {
                        $('span', '.fas.fa-home', '')
                        $('span', '.m-1', 'Home')
                    })
            })
        })

        return this.elem
    }

    destroy() {
    }
}

export class PageNotFoundHandler {
    constructor() {
        ROUTER.route('404', [PageNotFoundHandler.handlePageNotFound])
    }

    static handlePageNotFound = () => {
        SCREEN.setView(new PageNotFoundView())
    }
}
