import {IsUserLogged} from '../models/user'
import {ROUTER, SCREEN} from '../app'
import {Weya as $, WeyaElement} from '../../../lib/weya/weya'
import {Loader} from "../components/loader"
import CACHE, {IsUserLoggedCache} from "../cache/cache"
import NETWORK from '../network'
import {handleNetworkError} from '../utils/redirect';
import {setTitle} from '../utils/document'
import {ScreenView} from '../screen_view'

class LoginView extends ScreenView {
    isUserLogged: IsUserLogged
    isUserLoggedCache: IsUserLoggedCache
    elem: WeyaElement
    loader: Loader
    token: string
    returnUrl: string

    constructor(token?: string, returnUrl: string = '/runs') {
        super()

        this.isUserLoggedCache = CACHE.getIsUserLogged()
        this.loader = new Loader(true)
        this.token = token
        this.returnUrl = returnUrl
    }

    get requiresAuth(): boolean {
        return false;
    }

    render() {
        this.elem = $('div', $ => {
            this.loader.render($)
        })

        this.handleLogin().then()
        setTitle({section: 'Login'})

        return this.elem
    }

    private async handleLogin() {
        this.isUserLogged = await this.isUserLoggedCache.get()

        if (this.token) {
            try {
                let res = await NETWORK.signIn(this.token)
                if (!res.is_successful) {
                    ROUTER.navigate('/401')
                }
                localStorage.setItem('app_token', res.app_token)
                this.isUserLogged = await this.isUserLoggedCache.get(true)
            } catch (e) {
                handleNetworkError(e)
                return
            }
            await SCREEN.updateTheme().then()
        }

        if (!this.isUserLogged.is_user_logged) {
            NETWORK.redirectLogin()
            return
        }

        ROUTER.navigate(this.returnUrl)
        this.loader.remove()
    }
}

export class LoginHandler {
    constructor() {
        ROUTER.route('login', [this.handleLogin])
    }

    handleLogin = () => {
        let params = (window.location.hash.substr(1)).split("&")
        let token = undefined
        let returnUrl = sessionStorage.getItem('return_url') || '/runs'

        for (let i = 0; i < params.length; i++) {
            let val = params[i].split('=')
            switch (val[0]) {
                case 'access_token':
                    token = val[1]
                    break
                case 'return_url':
                    returnUrl = val[1]
                    break
            }
        }
        sessionStorage.setItem('return_url', returnUrl)
        SCREEN.setView(new LoginView(token, returnUrl))
    }
}
