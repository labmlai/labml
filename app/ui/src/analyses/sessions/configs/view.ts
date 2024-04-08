import {ROUTER, SCREEN} from "../../../app"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {Session} from "../../../models/session"
import {Status} from "../../../models/status"
import CACHE, {SessionCache, SessionStatusCache} from "../../../cache/cache"
import {DataLoader} from "../../../components/loader"
import {BackButton} from "../../../components/buttons"
import {SessionHeaderCard} from "../session_header/card"
import {Configs} from "./components"
import {ViewHandler} from "../../types"
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {setTitle} from '../../../utils/document'
import {ScreenView} from '../../../screen_view'

class SessionConfigsView extends ScreenView {
    elem: HTMLDivElement
    uuid: string
    session: Session
    status: Status
    statusCache: SessionStatusCache
    sessionCache: SessionCache
    actualWidth: number
    sessionHeaderCard: SessionHeaderCard
    configsContainer: HTMLDivElement
    private loader: DataLoader
    private refresh: AwesomeRefreshButton

    constructor(uuid: string) {
        super()

        this.uuid = uuid
        this.sessionCache = CACHE.getSession(this.uuid)
        this.statusCache = CACHE.getSessionStatus(this.uuid)

        this.loader = new DataLoader(async (force) => {
            this.status = await this.statusCache.get(force)
            this.session = await this.sessionCache.get(force)
        })
        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))

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
        setTitle({section: 'Configurations'})
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', '.page',
                {style: {width: `${this.actualWidth}px`}}, $ => {
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
                        $('h2', '.header.text-center', 'Configurations')
                        this.loader.render($)
                        this.configsContainer = $('div', '.labml-card')
                    })
                })
        })

        try {
            await this.loader.load()

            setTitle({section: 'Configurations', item: this.session.name})
            this.renderConfigsView()
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
            this.renderConfigsView()
        } catch (e) {

        } finally {
            if (this.status && !this.status.isRunning) {
                this.refresh.stop()
            }
            this.sessionHeaderCard.refresh().then()
        }
    }

    onVisibilityChange() {
        this.refresh.changeVisibility(!document.hidden)
    }

    renderConfigsView() {
        this.configsContainer.innerHTML = ''
        $(this.configsContainer, $ => {
            new Configs({configs: this.session.configs, width: this.actualWidth}).render($)
        })
    }

}

export class SessionConfigsHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('session/:uuid/configs', [this.handleSessionConfigs])
    }

    handleSessionConfigs = (uuid: string) => {
        SCREEN.setView(new SessionConfigsView(uuid))
    }
}
