import {LogModel, Logs, Run} from "../../../models/run"
import CACHE, {RunCache, RunStatusCache} from "../../../cache/cache"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import Filter from "../../../utils/ansi_to_html"
import {Status} from "../../../models/status"
import {ROUTER, SCREEN} from "../../../app"
import {BackButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import {DataLoader} from "../../../components/loader"
import {ViewHandler} from "../../types"
import {RefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {setTitle} from '../../../utils/document'
import {ScreenView} from '../../../screen_view'
import stdErrCache from "./cache"
import {LogView} from "../../../components/log_view"

class StdErrorView extends ScreenView {
    private elem: HTMLDivElement
    private readonly uuid: string
    private run: Run
    private stdErr: Logs
    private status: Status
    private statusCache: RunStatusCache
    private runCache: RunCache
    private actualWidth: number
    private outputContainer: HTMLDivElement
    private runHeaderCard: RunHeaderCard
    private filter: Filter
    private loader: DataLoader
    private refresh: RefreshButton
    private logView: LogView

    constructor(uuid: string) {
        super()

        this.uuid = uuid
        this.runCache = CACHE.getRun(this.uuid)
        this.statusCache = CACHE.getRunStatus(this.uuid)
        this.filter = new Filter({})

        this.loader = new DataLoader(async (force) => {
            this.status = await this.statusCache.get(force)
            this.run = await this.runCache.get(force)
            this.stdErr = await stdErrCache.getLogCache(this.uuid).getLast(force)
        })
        this.refresh = new RefreshButton(this.onRefresh.bind(this))
        this.logView = new LogView(new Logs(<LogModel>{pages: {}, page_length: 0}), async (currentPage): Promise<Logs> => {
            return await stdErrCache.getLogCache(this.uuid).getPage(currentPage, false)
        }, async (wrap: boolean): Promise<boolean> => {
            return await stdErrCache.getLogCache(this.uuid).updateLogWrap(wrap)
        })
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
        setTitle({section: 'Standard Error'})
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
                        width: this.actualWidth,
                        showRank: false
                    })
                    this.runHeaderCard.render($).then()
                    $('h2', '.header.text-center', 'Standard Error')
                    this.loader.render($)
                    this.outputContainer = $('div', '.terminal-card' , $ => {
                        this.logView.render($)
                    })
                })
            })
        })

        try {
            await this.loader.load()

            setTitle({section: 'Standard Error', item: this.run.name})
            this.renderOutput()
        } catch (e) {
            handleNetworkErrorInplace(e)
        }
    }

    render(): WeyaElement {
        this.elem = $('div')

        this._render().then()

        return this.elem
    }

    destroy() {

    }

    async onRefresh() {
        try {
            stdErrCache.getLogCache(this.uuid).invalidate_cache()
            this.logView.invalidateLogs()

            await this.loader.load(true)

            this.renderOutput()
        } catch (e) {
            handleNetworkErrorInplace(e)
        } finally {
            await this.runHeaderCard.refresh().then()
        }
    }

    onVisibilityChange() {

    }

    renderOutput() {
        this.logView.addLogs(this.stdErr)
    }
}

export class StdErrorHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('run/:uuid/stderr', [this.handleStdError])
    }

    handleStdError = (uuid: string) => {
        SCREEN.setView(new StdErrorView(uuid))
    }
}
