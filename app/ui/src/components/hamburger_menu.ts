import {Weya as $, WeyaElementFunction} from '../../../lib/weya/weya'
import {MenuButton, NavButton} from './buttons'
import {Loader} from './loader'
import CACHE, {UserCache} from "../cache/cache"
import {User} from '../models/user'
import NETWORK from '../network'
import {handleNetworkError} from '../utils/redirect';
import {Sentry} from '../sentry';
import {ROUTER} from '../app'
import {getPath} from '../utils/document'

const DEFAULT_IMAGE = 'https://raw.githubusercontent.com/azouaoui-med/pro-sidebar-template/gh-pages/src/img/user.jpg'

export interface HamburgerMenuOptions {
    title: string
    setButtonContainer?: (container: HTMLDivElement) => void
}

export class HamburgerMenuView {
    elem: HTMLDivElement
    navLinksContainer: HTMLElement
    overlayElement: HTMLDivElement
    buttonContainer: HTMLDivElement
    loader: Loader
    userCache: UserCache
    user: User
    isMenuVisible: boolean
    title: string
    setButtonContainer?: (container: HTMLDivElement) => void

    constructor(opt: HamburgerMenuOptions) {
        this.userCache = CACHE.getUser()

        this.title = opt.title
        this.setButtonContainer = opt.setButtonContainer

        this.loader = new Loader()
        this.isMenuVisible = false
    }

    render($: WeyaElementFunction) {
        this.elem = $('div', $ => {
            $('div', '.nav-container', $ => {
                this.navLinksContainer = $('nav', '.nav-links', $ => {
                    this.loader.render($)
                })
                new MenuButton({onButtonClick: this.onMenuToggle, parent: this.constructor.name}).render($)
                $('div', '.title', $ => {
                    $('h5', this.title)
                })
                this.buttonContainer = $('div', '.buttons', '')
            })
            this.overlayElement = $('div', '.overlay', {on: {click: this.onMenuToggle}})
        })

        this.renderProfile().then()

        if (this.setButtonContainer) {
            this.setButtonContainer(this.buttonContainer)
        }
        return this.elem
    }

    onMenuToggle = () => {
        this.isMenuVisible = !this.isMenuVisible
        if (this.isMenuVisible) {
            this.navLinksContainer.classList.add('nav-active')
            this.overlayElement.classList.add('d-block')
        } else {
            this.navLinksContainer.classList.remove('nav-active')
            this.overlayElement.classList.remove('d-block')
        }
    }

    onLogOut = () => {
        ROUTER.navigate(`/auth/sign_out?redirect_url=${encodeURIComponent(getPath())}`)
    }

    private async renderProfile() {
        try {
            this.user = await this.userCache.get()
        } catch (e) {
            //Do nothing since the error is handled by the parent view
        }

        this.loader.remove()

        $(this.navLinksContainer, $ => {
            $('div', '.text-center', $ => {
                $('img', '.mt-2.image-style.rounded-circle', {
                    src: this.user?.picture || DEFAULT_IMAGE
                })
                $('div', '.mb-5.mt-3.mt-2', $ => {
                    $('h5', this.user?.name || '')
                })
            })
            new NavButton({
                icon: '.fas.fa-running',
                text: 'Runs',
                link: '/runs',
                parent: this.constructor.name
            }).render($)
            new NavButton({
                icon: '.fas.fa-desktop',
                text: 'Computers',
                link: '/computers',
                parent: this.constructor.name
            }).render($)
            new NavButton({
                icon: '.fas.fa-book',
                text: 'Documentation',
                link: 'https://docs.labml.ai',
                target: '_blank',
                parent: this.constructor.name
            }).render($)
            new NavButton({
                icon: '.fas.fa-sliders-h',
                text: 'Settings',
                link: '/settings',
                parent: this.constructor.name
            }).render($)
            $('span', '.mt-5', '')
            new NavButton({
                icon: '.fas.fa-power-off',
                text: 'Sign out',
                onButtonClick: this.onLogOut,
                parent: this.constructor.name
            }).render($)
        })
    }
}
