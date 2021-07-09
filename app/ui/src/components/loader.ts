import {Weya, WeyaElement, WeyaElementFunction} from '../../../lib/weya/weya'
import {ErrorMessage} from './error_message'
import {waitForFrame} from '../utils/render'

export class Loader {
    elem: WeyaElement
    isScreenLoader: boolean

    constructor(isScreenLoader?: boolean) {
        this.isScreenLoader = isScreenLoader
        this.elem = null
    }

    render($: WeyaElementFunction) {
        if (this.isScreenLoader) {
            this.elem = $('div', '.loader-container', $ => {
                $('div', '.text-center.mt-5', $ => {
                    $('img', '.logo-style', {src: '/images/lab_logo.png'})
                })
                $('div', '.text-center', $ => {
                    $('div.loader', '')
                })
            })
        } else {
            this.elem = $('div', '.text-center', $ => {
                $('div', '.loader', '')
            })
        }

        return this.elem
    }

    remove() {
        if (this.elem == null) {
            return
        }
        this.elem.remove()
        this.elem = null
    }
}

export class DataLoader {
    private _load: (force: boolean) => Promise<void>
    private loaded: boolean
    private loader: Loader
    private elem: HTMLDivElement
    private errorMessage: ErrorMessage

    constructor(load: (force: boolean) => Promise<void>) {
        this._load = load
        this.loaded = false
        this.loader = new Loader()
        this.errorMessage = new ErrorMessage()
    }

    render($: WeyaElementFunction) {
        this.elem = $('div', '.data-loader')
    }

    async load(force: boolean = false) {
        this.errorMessage.remove()
        if (!this.loaded) {
            this.elem.appendChild(this.loader.render(Weya))
            await waitForFrame()
        }

        try {
            await this._load(force)
            this.loaded = true
        } catch (e) {
            this.loaded = false
            this.errorMessage.render(this.elem)
            throw e
        } finally {
            this.loader.remove()
        }
    }
}
