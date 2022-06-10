import {User} from '../../models/user'
import {ROUTER, SCREEN} from '../../app'
import {Weya as $} from '../../../../lib/weya/weya'
import {DataLoader} from "../../components/loader"
import CACHE, {UserCache} from "../../cache/cache"
import {handleNetworkErrorInplace} from '../../utils/redirect';
import {clearChildElements, setTitle} from '../../utils/document'
import {ScreenView} from '../../screen_view'
import {HamburgerMenuView} from '../../components/hamburger_menu'

class SignOutView extends ScreenView {
    elem: HTMLDivElement
    private loader: DataLoader
    private returnUrl: string
    private userCache: UserCache
    private user: User
    private actualWidth: number
    private loginContainer: HTMLDivElement

    constructor(returnUrl: string = '/runs') {
        super()

        this.userCache = CACHE.getUser()
        this.loader = new DataLoader(async (force: boolean) => {
            this.user = await this.userCache.get(force)
        })
        this.returnUrl = returnUrl
    }

    get requiresAuth(): boolean {
        return false;
    }

    onResize(width: number) {
        super.onResize(width)

        this.actualWidth = Math.min(400, width)

        if (this.elem) {
            this._render().then()
        }
    }

    render() {
        this.elem = $('div')

        this._render().then()

        return this.elem
    }

    async _render() {
        setTitle({section: `Sign Out`})
        clearChildElements(this.elem)
        $(this.elem, $ => {
            $('div', '.page', {style: {width: `${this.actualWidth}px`}}, $ => {
                $('div', $ => {
                    new HamburgerMenuView({
                        title: `Sign Out`,
                    }).render($)
                })

                this.loader.render($)

                this.loginContainer = $('div', '.auth-container')
            })
        })

        try {
            await this.loader.load(false)

            this.renderLogin()
        } catch (e) {
            handleNetworkErrorInplace(e)
            return
        }
    }

    renderLogin() {
        if (!this.user.is_complete) {
            ROUTER.navigate(this.returnUrl)
            return
        }

        this.userCache.signOut().then(value => {
            if (!value) {
                alert("Sign Out failed!")
            }
            ROUTER.navigate(this.returnUrl)
        })
    }
}

export class SignOutHandler {
    constructor() {
        ROUTER.route('auth/sign_out', [this.handleSignOut])
    }

    handleSignOut = () => {
        let urlParams = new URLSearchParams(window.location.search)
        let redirectURL = decodeURIComponent(urlParams.get('return_url') ?? '/')
        SCREEN.setView(new SignOutView(redirectURL))
    }
}
