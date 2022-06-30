import {User} from '../../models/user'
import {ROUTER, SCREEN} from '../../app'
import {Weya as $} from '../../../../lib/weya/weya'
import {DataLoader} from "../../components/loader"
import CACHE, {UserCache} from "../../cache/cache"
import {handleNetworkErrorInplace} from '../../utils/redirect';
import {clearChildElements, getPath, setTitle} from '../../utils/document'
import {ScreenView} from '../../screen_view'
import {HamburgerMenuView} from '../../components/hamburger_menu'
import EditableField from '../../components/input/editable_field'
import {CustomButton} from '../../components/buttons'
import {valueOrDefault} from '../../utils/value'

class SignUpView extends ScreenView {
    private elem: HTMLDivElement
    private loader: DataLoader
    private actualWidth: number
    private userCache: UserCache
    private user: User
    private returnUrl: string
    private signUpContainer: HTMLDivElement
    private nameField: EditableField
    private emailField: EditableField
    private passwordField: EditableField
    private signUpButton: CustomButton
    private submitButton: HTMLInputElement

    constructor(returnUrl: string = '/runs') {
        super()

        this.userCache = CACHE.getUser()
        this.returnUrl = returnUrl
        this.loader = new DataLoader(async (force: boolean) => {
            this.user = await this.userCache.get(force)
        })
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
        setTitle({section: `Sign Up`})
        clearChildElements(this.elem)
        $(this.elem, $ => {
            $('div', '.page', {style: {width: `${this.actualWidth}px`}}, $ => {
                $('div', $ => {
                    new HamburgerMenuView({
                        title: `Sign Up`,
                    }).render($)
                })

                this.loader.render($)

                $('form', {
                    on: {submit: this.onSubmit},
                }, $ => {
                    this.signUpContainer = $('div', '.auth-container')
                })
            })
        })

        try {
            await this.loader.load(false)

            this.renderSignUp()
        } catch (e) {
            handleNetworkErrorInplace(e)
            return
        }
    }

    renderSignUp() {
        if (this.user.is_complete) {
            ROUTER.navigate(this.returnUrl, {replace: true})
            return
        }

        clearChildElements(this.signUpContainer)
        $(this.signUpContainer, $ => {
            $('div', '.input-list-container', $ => {
                $('ul', $ => {
                    this.nameField = new EditableField({
                        name: 'Name',
                        value: null,
                        isEditable: true,
                        autocomplete: 'name',
                        required: true,
                    })
                    this.nameField.render($)
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
                        autocomplete: 'new-password',
                        required: true,
                    })
                    this.passwordField.render($)
                    this.submitButton = $('input', {
                        type: 'submit', style: {
                            visibility: 'hidden',
                            position: 'absolute',
                            top: '-100px',
                        }
                    })
                })
            })
            this.signUpButton = new CustomButton({
                text: 'Sign Up',
                parent: this.constructor.name,
                onButtonClick: () => this.submitButton.click()
            })
            this.signUpButton.render($)
            $('div', '.footer', $ => {
                $('span', 'Already have an account? ')
                let linkElem = $('a', {href: '/auth/sign_in', on: {click: this.handleSignIn}})
                linkElem.textContent = 'Sign In'
            })
        })
    }

    async handleSignUp() {
        if (valueOrDefault(this.nameField.getInput(), '').length == 0) {
            window.alert("Enter Name!")
            return
        }
        if (valueOrDefault(this.emailField.getInput(), '').length == 0) {
            window.alert("Enter Email!")
            return
        }
        if (valueOrDefault(this.passwordField.getInput(), '').length == 0) {
            window.alert("Enter Password!")
            return
        }

        this.signUpButton.disabled = true
        let response = await this.userCache.signUp({
            name: this.nameField.getInput(),
            email: this.emailField.getInput(),
            password: this.passwordField.getInput(),
        })

        if (!response) {
            this.signUpButton.disabled = false
            return
        }

        ROUTER.navigate(this.returnUrl, {replace: true})
    }

    private onSubmit = (e: Event) => {
        e.preventDefault()
        e.stopPropagation()
        this.handleSignUp().then()
    }

    private handleSignIn(e: Event) {
        e.preventDefault()
        ROUTER.navigate(`/auth/sign_in?return_url=${encodeURIComponent(getPath())}`, {replace: true})
    }
}

export class SignUpHandler {
    constructor() {
        ROUTER.route('auth/sign_up', [this.handleSignUp])
    }

    handleSignUp = () => {
        let urlParams = new URLSearchParams(window.location.search)
        let redirectURL = decodeURIComponent(urlParams.get('return_url') ?? '/')
        SCREEN.setView(new SignUpView(redirectURL))
    }
}
