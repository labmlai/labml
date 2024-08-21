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
import {UserMessages} from "../components/user_messages"

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
            $('div.input-list-container', $ => {
                $('ul', $ => {
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
        this.updateRadio()
        try {
            await this.userCache.setUser(this.user)
            this.user.theme = this.radioDark.checked ? DARK : LIGHT
            await SCREEN.updateTheme()
        } catch (e) {
            UserMessages.shared.networkError(e, "Failed to save")
            this.updateRadio()
        }
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
