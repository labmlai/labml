import {WeyaElementFunction} from "../../../lib/weya/weya"

const AUTO_REFRESH_TIME = 2 * 60

export class AwesomeRefreshButton {
    private readonly _refresh: () => Promise<void>
    private handler?: () => void
    private refreshTimeout: number
    private lastVisibilityChange: number;
    private isActive: boolean
    private isPaused: boolean
    private isRefreshing: boolean
    private refreshButton: HTMLElement;
    private refreshIcon: HTMLSpanElement
    private remainingTimeElem: HTMLSpanElement
    private remainingTime: number
    private isDisabled: boolean

    constructor(refresh: () => Promise<void>) {
        this._refresh = refresh
        this.refreshTimeout = null
        this.isActive = false
        this.isDisabled = false
        this.isPaused = false
    }

    set disabled(value: boolean) {
        this.isDisabled = value

        if (this.refreshButton) {
            if (this.isDisabled) {
                this.refreshButton.classList.add('disabled')
                this._stop()
                return
            }
            this.refreshButton.classList.remove('disabled')
            this.start()
            this.onRefreshClick().then()
        }
    }

    render($: WeyaElementFunction) {
        this.refreshButton = $('nav', `.nav-link.tab.float-right.btn-refresh${this.isDisabled ? '.disabled' : ''}`,
            {on: {click: this.onRefreshClick.bind(this)}, style: {display: 'none'}},
            $ => {
                this.refreshIcon = $('span', '.fas.fa-sync', '')
                this.remainingTimeElem = $('span', '.time-remaining', '')
            })
    }

    async onRefreshClick() {
        clearTimeout(this.refreshTimeout)
        this.remainingTime = 0
        await this.procTimerUpdate()
    }

    start() {
        // Only reinitialize the timer if it's not a resize re-render
        if (!this.isActive) {
            this.remainingTime = AUTO_REFRESH_TIME
            this._stop()
            this.refreshTimeout = window.setTimeout(this.procTimerUpdate.bind(this), 1000)
        }
        this.refreshButton.style.display = null
        this.isActive = true
    }

    _stop() {
        if (this.refreshTimeout != null) {
            clearTimeout(this.refreshTimeout)
            this.refreshTimeout = null
        }
    }

    stop() {
        this.isActive = false
        if (this.refreshButton != null) {
            this.refreshButton.style.display = 'none'
            this.remainingTimeElem.innerText = ''
        }
        this.handler = null
        this._stop()
    }

    pause() {
        if (this.isActive) {
            this._stop()
            this.refreshButton.classList.add('disabled')
            this.remainingTimeElem.innerText = ''
            this.isPaused = true
            this.isActive = false
        }
    }

    resume() {
        if (this.isPaused) {
            this.refreshButton.classList.remove('disabled')
            this.start()
            this.isPaused = false
        }
    }

    changeVisibility(isVisible: boolean) {
        let currentTime = Date.now()
        if (!isVisible) {
            this.lastVisibilityChange = currentTime
            this._stop()
            return
        }

        if (!this.isActive) {
            return
        }

        this.remainingTime = Math.floor(Math.max(0, (this.lastVisibilityChange + this.remainingTime * 1000) - currentTime) / 1000)
        this._stop()
        this.refreshTimeout = window.setTimeout(this.procTimerUpdate.bind(this), 1000)
    }

    attachHandler(handler?: () => void) {
        this.handler = handler
    }

    private procTimerUpdate = async () => {
        if (this.remainingTime > 0) {
            if (!this.isPaused) {
                this.remainingTimeElem.innerText = String(this.remainingTime--)
            }
        } else {
            this.remainingTimeElem.innerText = ''
            if (!this.isRefreshing) {
                this.isRefreshing = true
                await this.refresh()
                this.remainingTime = AUTO_REFRESH_TIME
                this.isRefreshing = false
            }
        }
        if (this.handler) {
            this.handler()
        }
        if (!this.isPaused) {
            this._stop()
            this.refreshTimeout = window.setTimeout(this.procTimerUpdate.bind(this), 1000)
        }
    }

    private refresh = async () => {
        this.refreshIcon.classList.add('spin')
        this.refreshButton.classList.add('disabled')
        await this._refresh()
        this.refreshIcon.classList.remove('spin')
        if (!this.isPaused) {
            this.refreshButton.classList.remove('disabled')
        }
    }
}
