import {ROUTER, SCREEN} from '../../app'
import {Weya as $} from '../../../../lib/weya/weya'
import {setTitle} from '../../utils/document'
import {ScreenView} from '../../screen_view'
import {NetworkError} from "../../network"

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

export class MiscErrorView extends ScreenView {
    elem: HTMLDivElement
    error?: NetworkError | Error

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

    constructor(error: NetworkError | Error = null) {
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
        setTitle({section: '500'})
        this.elem = $('div', '.error-container', $ => {
            $('h2', '.mt-5', 'Ooops! Something went wrong' + '')

            if (this.error != null && this.error instanceof NetworkError) {
                $('h1', `${this.error.statusCode}` ?? '500')
                $('p', 'Seems like we are having issues right now with the server' + '')
                $('div.code-sample.bg-dark.px-1.py-2.my-3', $ => {
                    if (this.error instanceof NetworkError) {
                       if (this.error.errorDescription != null) {
                        $('pre.text-white', this.error.errorDescription)
                        }
                        if (this.error.stackTrace != null) {
                            $('pre.text-white', this.error.stackTrace)
                        }
                    }
                })
            } else  if (this.error != null && this.error instanceof Error) {
                $('p', 'Seems like we are having issues right now' + '')
                $('div.code-sample.bg-dark.px-1.py-2.my-3', $ => {
                    $('pre.text-white', this.error.toString())
                })
            }

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

export class MiscErrorHandler {
    constructor() {
        ROUTER.route('500', [MiscErrorHandler.handleMiscError])
    }

    static handleMiscError = (error: NetworkError | Error = null) => {
        SCREEN.setView(new MiscErrorView(error))
    }
}
