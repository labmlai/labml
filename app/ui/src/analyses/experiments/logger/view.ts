import {Logs, Run} from "../../../models/run"
import CACHE, {RunCache, RunStatusCache} from "../../../cache/cache"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import Filter from "../../../utils/ansi_to_html"
import {Status} from "../../../models/status"
import {ROUTER, SCREEN} from "../../../app"
import {BackButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import {DataLoader} from "../../../components/loader"
import {ViewHandler} from "../../types"
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {setTitle} from '../../../utils/document'
import {ScreenView} from '../../../screen_view'
import stdLoggerCache from "./cache"

class LoggerView extends ScreenView {
    elem: HTMLDivElement
    uuid: string
    run: Run
    stdLogger: Logs
    status: Status
    statusCache: RunStatusCache
    runCache: RunCache
    actualWidth: number
    outputContainer: HTMLDivElement
    runHeaderCard: RunHeaderCard
    filter: Filter
    private loader: DataLoader
    private refresh: AwesomeRefreshButton

    constructor(uuid: string) {
        super()

        this.uuid = uuid
        this.runCache = CACHE.getRun(this.uuid)
        this.statusCache = CACHE.getRunStatus(this.uuid)
        this.filter = new Filter({})

        this.loader = new DataLoader(async (force) => {
            this.status = await this.statusCache.get(force)
            this.run = await this.runCache.get(force)
            this.stdLogger = await stdLoggerCache.getLogCache(this.uuid).get(force)
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
        setTitle({section: 'Logger'})
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', '.page', $ => {
                $('div', $ => {
                    $('div', '.nav-container', $ => {
                        new BackButton({text: 'Run', parent: this.constructor.name}).render($)
                        this.refresh.render($)
                    })
                    this.runHeaderCard = new RunHeaderCard({
                        uuid: this.uuid,
                        width: this.actualWidth
                    })
                    this.runHeaderCard.render($).then()
                    $('h2', '.header.text-center', 'Logger')
                    this.loader.render($)
                    this.outputContainer = $('div', '.terminal-card')
                })
            })
        })

        try {
            await this.loader.load()

            setTitle({section: 'Logger', item: this.run.name})
            this.renderOutput()
        } catch (e) {
            handleNetworkErrorInplace(e)
        } finally {
            if (this.status && this.status.isRunning) {
                this.refresh.attachHandler(this.runHeaderCard.renderLastRecorded.bind(this.runHeaderCard))
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

            this.renderOutput()
        } catch (e) {

        } finally {
            if (this.status && !this.status.isRunning) {
                this.refresh.stop()
            }

            await this.runHeaderCard.refresh().then()
        }
    }

    onVisibilityChange() {
        this.refresh.changeVisibility(!document.hidden)
    }

    renderOutput() {
        this.outputContainer.innerHTML = ''
        $(this.outputContainer, $ => {
            let output = $('pre', '')
            if (this.stdLogger) {
                output.innerHTML = this.filter.toHtml(this.stdLogger.logs)
            }
        })
    }
}

export class LoggerHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('run/:uuid/logger', [this.handleLogger])
    }

    handleLogger = (uuid: string) => {
        SCREEN.setView(new LoggerView(uuid))
    }
}
