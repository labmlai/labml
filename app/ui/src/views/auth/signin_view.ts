import {User} from '../../models/user'
import {ROUTER, SCREEN} from '../../app'
import {Weya as $} from '../../../../lib/weya/weya'
import {Loader} from "../../components/loader"
import CACHE, {UserCache} from "../../cache/cache"
import {handleNetworkErrorInplace} from '../../utils/redirect';
import {clearChildElements, getPath, setTitle} from '../../utils/document'
import {ScreenView} from '../../screen_view'
import {HamburgerMenuView} from '../../components/hamburger_menu'
import EditableField from '../../components/input/editable_field'
import {CustomButton} from '../../components/buttons'
import {valueOrDefault} from '../../utils/value'

class SignInView extends ScreenView {
    elem: HTMLDivElement
    loader: Loader
    token: string
    returnUrl: string
    private userCache: UserCache
    private user: User
    private actualWidth: number
    private menuContainer: HTMLDivElement
    private loginContainer: HTMLDivElement
    private emailField: EditableField
    private passwordField: EditableField
    private loginButton: CustomButton
    private submitButton: HTMLInputElement

    constructor(returnUrl: string = '/runs') {
        super()

        this.userCache = CACHE.getUser()
        this.loader = new Loader(true)
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
        setTitle({section: `Sign In`})
        clearChildElements(this.elem)
        $(this.elem, $ => {
            $('div', '.page', {style: {width: `${this.actualWidth}px`}}, $ => {
                this.menuContainer = $('div', $ => {
                    new HamburgerMenuView({
                        title: `Sign In`,
                    }).render($)
                })

                this.loader.render($)

                $('form', {
                    on: {submit: this.onSubmit},
                }, $ => {
                    this.loginContainer = $('div', '.auth-container')
                })
            })
        })

        try {
            this.user = await this.userCache.get()
            this.loader.remove()

            this.renderLogin()
        } catch (e) {
            handleNetworkErrorInplace(e)
            return
        }
    }

    renderLogin() {
        if (this.user.is_complete) {
            ROUTER.navigate(this.returnUrl)
            return
        }

        clearChildElements(this.loginContainer)
        $(this.loginContainer, $ => {
            $('div', '.input-list-container', $ => {
                $('ul', $ => {
                    this.emailField = new EditableField({
                        name: 'Email',
                        value: null,
                        isEditable: true,
                        type: 'email',
                        autocomplete: 'email',
                        required: true,
                    })
                    this.emailField.render($)
                    this.passwordField = new EditableField({
                        name: 'Password',
                        value: null,
                        isEditable: true,
                        type: 'password',
                        autocomplete: 'current-password',
                        required: true,
                    })
                    this.passwordField.render($)
                })
            })
            this.submitButton = $('input', {
                type: 'submit', style: {
                    visibility: 'hidden',
                    position: 'absolute',
                    top: '-100px',
                }
            })
            this.loginButton = new CustomButton({
                text: 'Sign In',
                parent: this.constructor.name,
                onButtonClick: () => this.submitButton.click()
            })
            this.loginButton.render($)
            $('div', '.footer', $ => {
                $('span', 'Not a member yet? ')
                let linkElem = $('a', {href: '/auth/sign_up', on: {click: this.handleSignUp}})
                linkElem.textContent = 'Sign Up'
            })
            $('div', '.footer', $ => {
                $('span', 'Forgot your password? ')
                $('a', 'Reset Password', {
                    href: 'mailto:contact@labml.ai?subject=Reset%20My%20Password%20on%20app.labml.ai&body=Please%20reset%20the%20password%20of%20my%20account%20associated%20with%20this%20email%20address',
                    on: {
                        click: this.sendPasswordResetEmail
                    }
                })
            })
        })
    }

    async handleLogin() {
        if (valueOrDefault(this.emailField.getInput(), '').length == 0) {
            window.alert("Enter Email!")
            return
        }
        if (valueOrDefault(this.passwordField.getInput(), '').length == 0) {
            window.alert("Enter Password!")
            return
        }

        this.loginButton.disabled = true
        let response = await this.userCache.signIn({
            email: this.emailField.getInput(),
            password: this.passwordField.getInput()
        })

        if (!response) {
            this.loginButton.disabled = false
            return
        }

        ROUTER.navigate(this.returnUrl)
    }

    private sendPasswordResetEmail = (evt: Event) => {
        evt.preventDefault()
        evt.stopPropagation()
        window.open(`mailto:contact@labml.ai?subject=Reset%20My%20Password%20on%20app.labml.ai&body=Please%20reset%20the%20password%20of%20my%20account%20associated%20with%20${this.emailField.getInput()}`)

    }

    private onSubmit = (e: Event) => {
        e.preventDefault()
        e.stopPropagation()
        this.handleLogin().then()
    }

    private handleSignUp(e: Event) {
        e.preventDefault()
        ROUTER.navigate(`/auth/sign_up?return_url=${encodeURIComponent(getPath())}`)
    }
}

export class SignInHandler {
    constructor() {
        ROUTER.route('auth/sign_in', [this.handleSignIn])
    }

    handleSignIn = () => {
        let urlParams = new URLSearchParams(window.location.search)
        let redirectURL = decodeURIComponent(urlParams.get('return_url') ?? '/')
        SCREEN.setView(new SignInView(redirectURL))
    }
}
