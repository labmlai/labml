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

class PasswordResetView extends ScreenView {
    private elem: HTMLDivElement
    private loader: DataLoader
    private actualWidth: number
    private userCache: UserCache
    private user: User
    private token: string
    private resetContainer: HTMLDivElement
    private passwordField: EditableField
    private passwordConfirmField: EditableField
    private resetButton: CustomButton
    private submitButton: HTMLInputElement

    constructor(token: string ) {
        super()

        this.userCache = CACHE.getUser()
        this.token = token
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
        if (valueOrDefault(this.token, '').length == 0) {
            window.alert('Invalid reset token. Please retry with the link you received.')
            ROUTER.navigate('/')
            return
        }
        setTitle({section: `Reset Password`})
        clearChildElements(this.elem)
        $(this.elem, $ => {
            $('div', '.page', {style: {width: `${this.actualWidth}px`}}, $ => {
                $('div', $ => {
                    new HamburgerMenuView({
                        title: `Reset Password`,
                    }).render($)
                })

                this.loader.render($)

                $('form', {
                    on: {submit: this.onSubmit},
                }, $ => {
                    this.resetContainer = $('div', '.auth-container')
                })
            })
        })

        try {
            await this.loader.load(false)

            this.renderResetPassword()
        } catch (e) {
            handleNetworkErrorInplace(e)
            return
        }
    }

    renderResetPassword() {
        if (this.user.is_complete) {
            window.alert('Please logout from your account before attempting a password reset.')
            ROUTER.navigate('/')
            return
        }

        clearChildElements(this.resetContainer)
        $(this.resetContainer, $ => {
            $('div', '.input-list-container', $ => {
                $('ul', $ => {
                    this.passwordField = new EditableField({
                        name: 'Password',
                        value: null,
                        isEditable: true,
                        type: 'password',
                        autocomplete: 'new-password',
                        required: true,
                    })
                    this.passwordField.render($)
                    this.passwordConfirmField = new EditableField({
                        name: 'Confirm Password',
                        value: null,
                        isEditable: true,
                        type: 'password',
                        autocomplete: 'new-password',
                        required: true,
                    })
                    this.passwordConfirmField.render($)
                    this.submitButton = $('input', {
                        type: 'submit', style: {
                            visibility: 'hidden',
                            position: 'absolute',
                            top: '-100px',
                        }
                    })
                })
            })
            this.resetButton = new CustomButton({
                text: 'Reset',
                parent: this.constructor.name,
                onButtonClick: () => this.submitButton.click()
            })
            this.resetButton.render($)
        })
    }

    async handlePasswordReset() {
        if (valueOrDefault(this.passwordField.getInput(), '').length == 0) {
            window.alert('Enter new password!')
            return
        }
        if (valueOrDefault(this.passwordConfirmField.getInput(), '').length == 0) {
            window.alert('Enter confirm password!')
            return
        }
        if (this.passwordField.getInput() != this.passwordConfirmField.getInput()) {
            window.alert('Passwords do not match')
            return
        }

        this.resetButton.disabled = true
        let response = await this.userCache.resetPassword({
            reset_token: this.token,
            new_password: this.passwordField.getInput(),
        })

        if (!response) {
            this.resetButton.disabled = false
            return
        }

        window.alert('Password changed successfully. Please login back to your account.')
        ROUTER.navigate('/auth/sign_in')
    }

    private onSubmit = (e: Event) => {
        e.preventDefault()
        e.stopPropagation()
        this.handlePasswordReset().then()
    }
}

export class PasswordResetHandler {
    constructor() {
        ROUTER.route('auth/reset', [this.handlePasswordReset])
    }

    handlePasswordReset = () => {
        let urlParams = new URLSearchParams(window.location.search)
        let token = urlParams.get('token')
        SCREEN.setView(new PasswordResetView(token))
    }
}
