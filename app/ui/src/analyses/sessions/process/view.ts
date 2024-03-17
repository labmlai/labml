import {Process} from "./types"
import CACHE, {ProcessDataCache, SessionCache, SessionStatusCache} from "../../../cache/cache"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {Status} from "../../../models/status"
import {DataLoader} from "../../../components/loader"
import {ROUTER, SCREEN} from "../../../app"
import {BackButton} from "../../../components/buttons"
import {processCache} from "./cache"
import {SessionHeaderCard} from '../session_header/card'
import mix_panel from "../../../mix_panel"
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {ProcessList} from "./process_list"
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {Session} from '../../../models/session'
import {setTitle} from '../../../utils/document'
import {ScreenView} from '../../../screen_view'

class ProcessView extends ScreenView {
    elem: HTMLDivElement
    uuid: string
    status: Status
    actualWidth: number
    statusCache: SessionStatusCache
    series: Process[]
    analysisCache: ProcessDataCache
    sessionHeaderCard: SessionHeaderCard
    processListContainer: HTMLDivElement
    private loader: DataLoader
    private refresh: AwesomeRefreshButton
    private sessionCache: SessionCache
    private session: Session

    constructor(uuid: string) {
        super()

        this.uuid = uuid
        this.sessionCache = CACHE.getSession(this.uuid)
        this.statusCache = CACHE.getSessionStatus(this.uuid)
        this.analysisCache = processCache.getAnalysis(this.uuid)

        this.loader = new DataLoader(async (force) => {
            this.status = await this.statusCache.get(force)
            this.session = await this.sessionCache.get()
            this.series = (await this.analysisCache.get(force)).series
        })
        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))

        mix_panel.track('Analysis View', {uuid: this.uuid, analysis: this.constructor.name})
    }

    get requiresAuth(): boolean {
        return false
    }

    onResize(width: number) {
        super.onResize(width)

        this.actualWidth = Math.min(800, width)

        if (this.elem) {
            this._render().then()
        }
    }

    async _render() {
        setTitle({section: 'Processes'})
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', '.page',
                {style: {width: `${this.actualWidth}px`}},
                $ => {
                    $('div', $ => {
                        $('div', '.nav-container', $ => {
                            new BackButton({text: 'Session', parent: this.constructor.name}).render($)
                            this.refresh.render($)
                        })
                        this.sessionHeaderCard = new SessionHeaderCard({
                            uuid: this.uuid,
                            width: this.actualWidth
                        })
                        this.sessionHeaderCard.render($).then()
                        $('h2', '.header.text-center', 'Processes')
                        this.loader.render($)
                        this.processListContainer = $('div')
                    })
                })
        })

        try {
            await this.loader.load()

            setTitle({section: 'Processes', item: this.session.name})
            this.renderProcessList()
        } catch (e) {
            handleNetworkErrorInplace(e)
        } finally {
            if (this.status && this.status.isRunning) {
                this.refresh.attachHandler(this.sessionHeaderCard.renderLastRecorded.bind(this.sessionHeaderCard))
                this.refresh.start()
            }
        }
    }

    render(): WeyaElement {
        this.elem = $('div')

        this._render().then()

        return this.elem
    }

    destroy() {
        this.refresh.stop()
    }

    async onRefresh() {
        try {
            await this.loader.load(true)
            this.renderProcessList()
        } catch (e) {

        } finally {
            if (this.status && !this.status.isRunning) {
                this.refresh.stop()
            }
            await this.sessionHeaderCard.refresh().then()
        }
    }

    onVisibilityChange() {
        this.refresh.changeVisibility(!document.hidden)
    }

    renderProcessList() {
        this.processListContainer.innerHTML = ''
        $(this.processListContainer, $ => {
            new ProcessList({items: this.series, width: this.actualWidth, uuid: this.uuid}).render($)
        })
    }
}

export class ProcessHandler {
    constructor() {
        ROUTER.route('session/:uuid/process', [this.handleProcess])
    }

    handleProcess = (uuid: string) => {
        SCREEN.setView(new ProcessView(uuid))
    }
}
