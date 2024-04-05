import {ROUTER, SCREEN} from '../app'
import {Weya as $, WeyaElement} from '../../../lib/weya/weya'
import {Loader} from "../components/loader"
import CACHE, {UserCache} from "../cache/cache"
import {HamburgerMenuView} from '../components/hamburger_menu'
import {User} from '../models/user'
import EditableField from '../components/input/editable_field'
import {handleNetworkError} from '../utils/redirect'
import {setTitle} from '../utils/document'
import {ScreenView} from '../screen_view'

const DEFAULT_IMAGE = '/images/user.png'
const LIGHT = 'light'
const DARK = 'dark'

class SettingsView extends ScreenView {
    userCache: UserCache
    user: User
    elem: WeyaElement
    settingsContainer: WeyaElement
    loader: Loader
    actualWidth: number
    radioLight: HTMLInputElement
    radioDark: HTMLInputElement

    constructor() {
        super()

        this.userCache = CACHE.getUser()
        this.loader = new Loader(true)

    }

    onResize(width: number) {
        this.actualWidth = Math.min(800, width)
    }

    render() {
        this.elem = $('div', '.page', $ => {
            new HamburgerMenuView({title: 'Settings'}).render($)
            this.settingsContainer = $('div', '.auto-margin', {style: {width: `${this.actualWidth}px`}})
            this.loader.render($)
        })

        this.renderContent().then()

        return this.elem
    }

    private async renderContent() {
        setTitle({section: 'Settings'})
        try {
            this.user = await this.userCache.get()
        } catch (e) {
            handleNetworkError(e)
            return
        }

        this.loader.remove()

        $(this.settingsContainer, $ => {
            $('div', '.text-center', $ => {
                $('img', '.mt-2.image-style.rounded-circle', {
                    src: this.user.picture || DEFAULT_IMAGE
                })
            })
            $('div.input-list-container', $ => {
                $('ul', $ => {
                    new EditableField({
                        name: 'Token',
                        value: this.user.default_project
                    }).render($)
                    new EditableField({
                        name: 'Name',
                        value: this.user.name
                    }).render($)
                    new EditableField({
                        name: 'Email',
                        value: this.user.email
                    }).render($)
                    $(`li`, $ => {
                        $('span.item-key', 'Theme')
                        $('span.item-value', {on: {change: this.onThemeUpdate}}, $ => {
                            $('div', '.radio-button', $ => {
                                this.radioLight = $('input', '#l-option', {type: 'radio', name: 'selector'})
                                $('label', '.ms-2.mb-2', 'Light', {for: 'l-option'})
                            })
                            $('div', '.radio-button', $ => {
                                this.radioDark = $('input', '#d-option', {
                                    type: 'radio',
                                    name: 'selector',
                                    checked: null
                                })
                                $('label', '.ms-2.mb-2', 'Dark', {for: 'd-option'})
                            })
                        })
                    })
                })
            })
        })
        this.updateRadio()
    }

    onThemeUpdate = async () => {
        this.user.theme = this.radioDark.checked ? DARK : LIGHT
        this.userCache.setUser(this.user).then()
        this.updateRadio()
        await SCREEN.updateTheme()
    }

    updateRadio = () => {
        this.radioLight.checked = this.user.theme === LIGHT
        this.radioDark.checked = this.user.theme === DARK
    }
}

export class SettingsHandler {
    constructor() {
        ROUTER.route('settings', [this.handleSettings])
    }

    handleSettings = () => {
        SCREEN.setView(new SettingsView())
    }
}
