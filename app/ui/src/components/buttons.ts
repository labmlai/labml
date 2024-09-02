import {Weya as $, WeyaElement, WeyaElementFunction} from "../../../lib/weya/weya"
import {ROUTER} from '../app'
import runHeaderAnalysis from '../analyses/experiments/run_header'

export interface ButtonOptions {
    onButtonClick?: () => void
    isDisabled?: boolean
    parent: string
    title?: string
}

export abstract class Button {
    onButtonClick: () => void
    isDisabled: boolean
    parent: string
    elem?: WeyaElement
    title?: string

    protected constructor(opt: ButtonOptions) {
        this.onButtonClick = opt.onButtonClick
        this.isDisabled = opt.isDisabled ? opt.isDisabled : false
        this.parent = opt.parent
        this.title = opt.title
    }

    set disabled(isDisabled: boolean) {
        this.isDisabled = isDisabled
        if (this.elem) {
            if (this.isDisabled) {
                this.elem.classList.add('disabled')
                return
            }
            this.elem.classList.remove('disabled')
        }
    }

    onClick = (e: Event) => {
        e.preventDefault()
        if (!this.isDisabled) {
            this.onButtonClick()
        }
    }

    hide = (isHidden: boolean) => {
        if (this.elem == null) {
            return
        }
        if (isHidden) {
            this.elem.classList.add('hide')
        } else {
            this.elem.classList.remove('hide')
        }
    }

    render($: WeyaElementFunction) {
    }
}

interface BackButtonOptions extends ButtonOptions {
    text: string
}

export class BackButton extends Button {
    currentPath: string
    navigatePath: string
    text: string = 'Home'

    constructor(opt: BackButtonOptions) {
        super(opt)

        this.text = opt.text
        this.currentPath = window.location.pathname
        this.navigatePath = null

        if (/\/run\/.+_\d+/.test(this.currentPath)) {
            this.navigatePath = this.currentPath.replace(/(\/run\/.+?)_\d+/, '$1');
            this.text = "Run"
        } else if (/\/run\/.+\/metrics\/.+/.test(this.currentPath)) {
            this.text = 'Run'
            this.navigatePath = this.currentPath.replace(/\/run\/(.+)\/metrics\/.+/, '/run/$1');
        } else if (/\/run\/.+\/.+/.test(this.currentPath)) {
            this.text = 'Run'
            this.navigatePath = this.currentPath.replace(/\/run\/(.+)\/.+/, '/run/$1')
        } else if (/\/session\/.+\/.+/.test(this.currentPath)) {
            this.text = 'Computer'
            if (window.location.search.includes('run')) {
                this.text = 'Run'
            }
            this.navigatePath = this.currentPath.replace(/\/session\/(.+)\/.+/, '/session/$1')
        } else if (this.currentPath.includes('run')) {
            this.text = 'Runs'
            this.navigatePath = 'runs'
        } else if (this.currentPath.includes('session')) {
            this.text = 'Computers'
            if (window.location.search.includes("run")) {
                this.text = "Run"
            }
            this.navigatePath = 'computers'
        } else if (this.currentPath.includes(runHeaderAnalysis.route)) {
            this.text = 'Run'
            this.navigatePath = this.currentPath.replace(runHeaderAnalysis.route, 'run')
        }
    }

    onClick = () => {
        if (ROUTER.canBack()) {
            ROUTER.back()
            return
        }
        ROUTER.navigate(this.navigatePath, {replace: true})
    }

    render($: WeyaElementFunction) {
        this.elem = $('nav', `.nav-link.tab.float-left${this.isDisabled ? '.disabled' : ''}`,
            {on: {click: this.onClick}},
            $ => {
                $('span', '.fas.fa-chevron-left', '')
                $('span', '.ms-1', this.text)
            })
    }
}

export class RefreshButton extends Button {

    constructor(opt: ButtonOptions) {
        super(opt)
    }

    render($: WeyaElementFunction) {
        this.elem = $('nav', `.nav-link.tab.float-right${this.isDisabled ? '.disabled' : ''}`,
            {on: {click: this.onClick}},
            $ => {
                $('span', '.fas.fa-sync', '')
            })
    }

    remove() {
        this.elem.remove()
    }
}

export class SaveButton extends Button {

    constructor(opt: ButtonOptions) {
        super(opt)
    }

    set loading(value: boolean) {
        this.elem.innerHTML = ''
        if (value) {
            $(this.elem, $ => {
                $('span', '.fa.fa-spinner.fa-spin', '')
            })
        } else {
            $(this.elem, $ => {
                $('span', '.fas.fa-save', '')
            })
        }
    }

    render($: WeyaElementFunction) {
        this.elem = $('nav', `.nav-link.tab.float-right${this.isDisabled ? '.disabled' : ''}`,
            {on: {click: this.onClick}},
            $ => {
                $('span', '.fas.fa-save', '')
            })
    }
}

export class AddButton extends Button {

    constructor(opt: ButtonOptions) {
        super(opt)
    }

    set loading(value: boolean) {
        this.elem.innerHTML = ''
        if (value) {
            $(this.elem, $ => {
                $('span', '.fa.fa-spinner.fa-spin', '')
            })
        } else {
            $(this.elem, $ => {
                $('span', '.fas.fa-plus', '')
            })
        }
    }

    render($: WeyaElementFunction) {
        this.elem = $('nav', `.nav-link.tab.float-right${this.isDisabled ? '.disabled' : ''}`,
            {on: {click: this.onClick}},
            $ => {
                $('span', '.fas.fa-plus', '')
            })
    }
}

export class ExpandButton extends Button {
    private isExpanded: Boolean
    private iconContainer: WeyaElement

    constructor(opt: ButtonOptions) {
        super(opt)
        this.isExpanded = false
    }

    private onExpand = (e: Event) => {
        this.isExpanded = !this.isExpanded
        this.renderIcon()
        this.onClick(e)
    }

    private renderIcon() {
        if (this.isExpanded) {
            this.iconContainer.classList.remove("fa-chevron-left")
            this.iconContainer.classList.add("fa-chevron-down")
        } else {
            this.iconContainer.classList.remove("fa-chevron-add")
            this.iconContainer.classList.add("fa-chevron-left")
        }
    }

    render($: WeyaElementFunction) {
        this.elem = $('nav', `.nav-link.tab.float-right${this.isDisabled ? '.disabled' : ''}`,
            {on: {click: this.onExpand}},
            $ => {
                this.iconContainer = $('span', this.isExpanded ? '.fas.fa-chevron-down' : '.fas.fa-chevron-left', '')
            })
    }
}

export class EditButton extends Button {
    constructor(opt: ButtonOptions) {
        super(opt)
    }

    render($: WeyaElementFunction) {
        this.elem = $('nav', `.nav-link.tab.float-right${this.isDisabled ? '.disabled' : ''}`,
            {on: {click: this.onClick}},
            $ => {
                $('span', '.fas.fa-edit', '')
            })
    }

    remove() {
        this.elem.remove()
    }
}

export class DeleteButton extends Button {
    constructor(opt: ButtonOptions) {
        super(opt)
    }

    set loading(value: boolean) {
        this.elem.innerHTML = ''
        if (value) {
            $(this.elem, $ => {
                $('span', '.fa.fa-spinner.fa-spin', '')
            })
        } else {
            $(this.elem, $ => {
                $('span', '.fas.fa-trash', '')
            })
        }
    }

    render($: WeyaElementFunction) {
        this.elem = $('nav', `.nav-link.tab.float-right.danger${this.isDisabled ? '.disabled' : ''}`,
            {on: {click: this.onClick}},
            $ => {
                $('span', '.fas.fa-trash', '')
            })
    }
}

export class IconButton extends Button {
    private readonly icon: string
    constructor(opt: ButtonOptions, icon: string) {
        super(opt)
        this.icon = icon
    }

    set loading(value: boolean) {
        this.elem.innerHTML = ''
        if (value) {
            $(this.elem, $ => {
                $('span', '.fa.fa-spinner.fa-spin', '')
            })
        } else {
            $(this.elem, $ => {
                $('span', this.icon, '')
            })
        }
    }

    render($: WeyaElementFunction) {
        this.elem = $('nav', `.nav-link.tab.float-right${this.isDisabled ? '.disabled' : ''}`,
            {on: {click: this.onClick}},
            $ => {
                $('span', this.icon, '')
            })
    }
}

export class CancelButton extends Button {
    constructor(opt: ButtonOptions) {
        super(opt)
    }

    render($: WeyaElementFunction) {
        this.elem = $('nav', `.nav-link.tab.float-right${this.isDisabled ? '.disabled' : ''}`,
            {on: {click: this.onClick}},
            $ => {
                $('span', '.fas.fa-times', '')
            })
    }
}

export class CleanButton extends Button {
    private refreshIcon: HTMLSpanElement

    constructor(opt: ButtonOptions) {
        super(opt)
    }

    set isLoading(value: boolean) {
        if (value) {
            this.refreshIcon.classList.remove('hide')
        } else {
            this.refreshIcon.classList.add('hide')
        }
    }

    render($: WeyaElementFunction) {
        this.elem = $('nav', `.nav-link.tab.float-right${this.isDisabled ? '.disabled' : ''}`,
            {on: {click: this.onClick}, title: 'clean checkpoints'},
            $ => {
                $('span', '.fas.fa-broom', '')
                this.refreshIcon = $('span', '.fas.fa-sync.spin', {style: {'margin-right': '6px'}})
            })
    }
}

export class MenuButton extends Button {
    constructor(opt: ButtonOptions) {
        super(opt)
    }

    render($: WeyaElementFunction) {
        this.elem = $('nav', `.burger.nav-link.tab.float-left${this.isDisabled ? '.disabled' : ''}`,
            {on: {click: this.onClick}},
            $ => {
                $('span', '.fas.fa-bars', '')
            })
    }
}

interface NavButtonOptions extends ButtonOptions {
    text: string
    icon: string
    link?: string
    target?: string
}

export class NavButton extends Button {
    text: string
    icon: string
    link?: string
    target?: string
    isDisabled: boolean

    constructor(opt: NavButtonOptions) {
        super(opt)
        this.icon = opt.icon
        this.text = opt.text
        this.link = opt.link
        this.target = opt.target
        this.isDisabled = opt.isDisabled ?? false
    }

    render($: WeyaElementFunction) {
        this.elem = $('a', '.nav-link.tab',
            {href: this.link, target: this.target, on: {click: this.onClick}},
            $ => {
                $('span', this.icon, '')
                $('span', '', this.text)
            })
        if (this.isDisabled) {
            this.elem.classList.add("disabled")
        }
    }

    onClick = (e: Event) => {
        e.preventDefault()
        if (this.link) {
            if (this.target === '_blank') {
                window.open(this.link)
                return
            }
            ROUTER.navigate(this.link)
            return
        }
        this.onButtonClick()
    }
}

interface ToggleButtonOptions extends ButtonOptions {
    text: string
    isToggled: boolean
}

export class ToggleButton extends Button {
    text: string
    isToggled: boolean
    defaultClass: string

    constructor(opt: ToggleButtonOptions) {
        super(opt)

        this.text = opt.text
        this.isToggled = opt.isToggled
        this.defaultClass = this.isToggled ? 'selected' : 'empty'
    }

    set toggle(isToggle: boolean) {
        this.isToggled = isToggle
        this.defaultClass = this.isToggled ? 'selected' : 'empty'
        this.renderToggleButton()
    }

    onClick = () => {
        this.onButtonClick()

        this.defaultClass = this.defaultClass === 'selected' ? 'empty' : 'selected'
        this.renderToggleButton()
    }

    render($: WeyaElementFunction) {
        this.elem = $('div', '.d-flex',
            {on: {click: this.onClick}}
        )
        this.defaultClass = this.isToggled ? 'selected' : 'empty'
        this.renderToggleButton()
    }

    renderToggleButton() {
        this.elem.innerHTML = ''

        $(this.elem, $ => {
            $('nav', `.nav-link.float-left.tab.toggle-button.${this.defaultClass}`,
                $ => {
                    $('span', this.text)
                }
            )
        })
    }
}

interface ControlledToggleButtonOptions extends ButtonOptions {
    text: string
    isToggled: boolean
    icon?: string
}

export class ControlledToggleButton extends Button {
    private readonly text: string
    private isToggled: boolean
    private readonly icon: string

    constructor(opt: ControlledToggleButtonOptions) {
        super(opt)

        this.text = opt.text
        this.isToggled = opt.isToggled
        this.icon = opt.icon ? opt.icon : ''
    }

    set toggle(isToggle: boolean) {
        this.isToggled = isToggle

        this.renderToggleButton()
    }

    onClick = () => {
        this.onButtonClick()
    }

    render($: WeyaElementFunction) {
        this.elem = $('span.toggle-button', '',
            {on: {click: this.onClick}}
        )

        this.renderToggleButton()
    }

    private renderToggleButton() {
        if (this.elem == null) {
            return
        }

        this.elem.innerHTML = ''

        $(this.elem, $ => {
            $('nav', `.nav-link.tab.toggle-button.${this.isToggled ? 'selected' : 'empty'}`,
                $ => {
                    if (this.icon) {
                        $('span', this.icon, '')
                    }
                    $('span', this.text)
                }
            )
        })
    }
}

interface CustomButtonOptions extends ButtonOptions {
    text: string
    noMargin?: boolean
}

export class CustomButton extends Button {
    text: string
    noMargin: boolean

    constructor(opt: CustomButtonOptions) {
        super(opt)
        this.text = opt.text
        this.noMargin = opt.noMargin ? opt.noMargin : false
    }

    render($: WeyaElementFunction) {
        this.elem = $('nav', `.nav-link${this.noMargin?'':'.mb-2'}.tab${this.isDisabled ? '.disabled' : ''}`,
            {on: {click: this.onClick}, title: this.title},
            $ => {
                $('span', this.text)
            })
    }
}

export class TensorBoardButton extends Button {
    private refreshIcon: HTMLSpanElement

    constructor(opt: ButtonOptions) {
        super(opt)
    }

    set isLoading(value: boolean) {
        if (value) {
            this.refreshIcon.classList.remove('hide')
        } else {
            this.refreshIcon.classList.add('hide')
        }
    }

    render($: WeyaElementFunction) {
        this.elem = $('nav', `.nav-link.tab${this.isDisabled ? '.disabled' : ''}`,
            {on: {click: this.onClick}, title: 'start TensorBoard', style: {padding: '2px 6px 2px 6px'}},
            $ => {
                $('img', {src: '../../images/tf_Icon.png', width: `${35}px`})
                this.refreshIcon = $('span', '.fas.fa-sync.spin', {style: {'margin-right': '6px'}})
            })
    }
}

interface ShareButtonOptions extends ButtonOptions {
    text: string
}

export class ShareButton extends Button {
    private toastDiv: HTMLDivElement

    constructor(opt: ShareButtonOptions) {
        super(opt)

        this._text = `Check this ${opt.text} on labml`
    }

    private _text: string

    set text(value: string) {
        this._text = `Check this ${value} on labml`
    }

    render($: WeyaElementFunction) {
        this.elem = $('nav', `.nav-link.tab.float-right${this.isDisabled ? '.disabled' : ''}`,
            {on: {click: this.onClick}},
            $ => {
                $('span', '.fas.fa-share-alt', '')
            })

        this.renderToast()

    }

    renderToast() {
        if (this.toastDiv != null) {
            this.toastDiv.remove()
        }
        $(document.body, $ => {
            $('div', '.toast-custom-container', $ => {
                this.toastDiv = $('div', '.toast.align-items-center', {
                    role: 'alert',
                    'aria-live': 'assertive',
                    'aria-atomic': 'true'
                }, $ => {
                    $('div', '.d-flex', $ => {
                        $('div', '.toast-body', 'Link copied!')
                    })
                })
            })
        })
    }

    copyLink() {
        this.toastDiv.classList.add('show')
        $(document.body, $ => {
            let elm = $('input', {value: window.location.href})
            elm.select()
            document.execCommand('copy')
            elm.remove()
        })
        window.setTimeout(() => {
            this.toastDiv.classList.remove('show')
        }, 2500)
    }

    onClick = async () => {
        if (navigator.share) {
            try {
                await navigator.share({
                    text: `${this._text}`,
                    url: window.location.href
                })
            } catch (e) {
                this.copyLink()
            }
        } else {
            this.copyLink()
        }
    }
}
