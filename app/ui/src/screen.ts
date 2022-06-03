import {Weya as $} from '../../lib/weya/weya'
import {getWindowDimensions} from "./utils/window_dimentions"
import CACHE, {UserCache} from './cache/cache'
import {Loader} from './components/loader'
import {ROUTER} from './app'
import {setTitle} from './utils/document'
import {ScreenView} from './screen_view'
import {handleNetworkErrorInplace} from './utils/redirect'

class ScreenContainer {
    view?: ScreenView
    private userCache: UserCache
    private loader: Loader
    private windowWidth: number

    constructor() {
        this.view = null
        this.userCache = CACHE.getUser()
        this.loader = new Loader(true)
        window.addEventListener('resize', this.onResize.bind(this))
        document.addEventListener('visibilitychange', this.onVisibilityChange.bind(this))
    }

    onResize = () => {
        let windowWidth = getWindowDimensions().width
        // Prevent mobile browser addressBar visibility from triggering a resize event
        if (this.windowWidth !== windowWidth && this.view) {
            this.windowWidth = windowWidth
            this.view.onResize(windowWidth)
        }
    }

    onVisibilityChange() {
        if (this.view) {
            this.view.onVisibilityChange()
        }
    }

    async updateTheme() {
        let theme = localStorage.getItem('theme') || 'light'
        if (document.body.className !== theme) {
            document.body.className = theme
        }
        try {
            let user = await this.userCache.get()
            if (user != null) {
                theme = user.theme
                localStorage.setItem('theme', theme)
            }
        } catch (e) {
            //Let the view handle network failures
        }
        if (document.body.className !== theme) {
            document.body.className = theme || 'light'
        }
    }

    setView(view: ScreenView) {
        if (this.view) {
            this.view.destroy()
            setTitle({})
        }
        this.view = view
        document.body.innerHTML = ''
        this.updateTheme().then()
        this.loader.render($)
        if (!this.view.requiresAuth) {
            document.body.innerHTML = ''
            this.windowWidth = null
            this.onResize()
            document.body.append(this.view.render())
            return
        }
        this.userCache.get().then(value => {
            if (this.view.requiresAuth && value != null) {
                ROUTER.navigate(`/login#return_url=${window.location.pathname}`)
                return
            }
            document.body.innerHTML = ''
            this.windowWidth = null
            this.onResize()
            document.body.append(this.view.render())
        }).catch(e => {
            handleNetworkErrorInplace(e)
        })
    }
}

export {ScreenContainer}
