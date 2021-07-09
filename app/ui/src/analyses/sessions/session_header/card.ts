import {Weya, WeyaElementFunction} from '../../../../../lib/weya/weya'
import CACHE, {SessionCache, SessionStatusCache} from "../../../cache/cache"
import {CardOptions} from "../../types"
import {Status} from "../../../models/status"
import {StatusView} from "../../../components/status"
import {formatTime, getTimeDiff} from "../../../utils/time"
import {Session} from '../../../models/session'
import {DataLoader} from '../../../components/loader'
import {ROUTER} from "../../../app"

interface SessionHeaderOptions extends CardOptions {
    lastUpdated?: number
    clickable?: boolean
}

export class SessionHeaderCard {
    session: Session
    uuid: string
    status: Status
    lastUpdated: number
    sessionCache: SessionCache
    elem: HTMLDivElement
    lastRecordedContainer: HTMLDivElement
    statusViewContainer: HTMLDivElement
    statusCache: SessionStatusCache
    private loader: DataLoader
    private clickable: boolean

    constructor(opt: SessionHeaderOptions) {
        this.uuid = opt.uuid
        this.clickable = opt.clickable ?? false
        this.lastUpdated = opt.lastUpdated
        this.sessionCache = CACHE.getSession(this.uuid)
        this.statusCache = CACHE.getSessionStatus(this.uuid)

        this.loader = new DataLoader(async (force) => {
            this.status = await this.statusCache.get(force)
            this.session = await this.sessionCache.get(force)

            this.lastUpdated = opt.lastUpdated ? opt.lastUpdated : this.statusCache.lastUpdated
        })
    }

    async render($: WeyaElementFunction) {
        this.elem = $('div', '.labml-card.disabled', $ => {
            this.loader.render($)
        })

        if (this.clickable) {
            this.elem.classList.remove('disabled')
            this.elem.classList.add('labml-card-action')
            this.elem.addEventListener('click', this.onClick)
        }

        try {
            await this.loader.load()

            Weya(this.elem, $ => {
                $('div', $ => {
                    $('div', $ => {
                        this.lastRecordedContainer = $('div', '.last-updated.mb-2')
                    })
                    $('div', '.run-info', $ => {
                        this.statusViewContainer = $('div')
                        $('h3', `${this.session.name}`)
                        $('h5', `${this.session.comment}`)
                    })
                })
            })

            this.renderStatusView()
            this.renderLastRecorded()
        } catch (e) {

        }

    }

    renderLastRecorded() {
        let lastRecorded = this.status.last_updated_time
        this.lastRecordedContainer.innerText = `Last Recorded ${this.status.isRunning ?
            getTimeDiff(lastRecorded * 1000) : formatTime(lastRecorded)}`
    }

    renderStatusView() {
        this.statusViewContainer.innerHTML = ''
        Weya(this.statusViewContainer, $ => {
            new StatusView({status: this.status.run_status, type: 'session'}).render($)
        })
    }

    async refresh(lastUpdated?: number) {
        try {
            await this.loader.load(true)

            this.lastUpdated = lastUpdated ? lastUpdated : this.statusCache.lastUpdated

            this.renderStatusView()
            this.renderLastRecorded()
        } catch (e) {

        }
    }

    onClick = () => {
        ROUTER.navigate(`/session/${this.uuid}/header`)
    }
}
