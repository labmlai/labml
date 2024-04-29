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
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {setTitle} from '../../../utils/document'
import {ScreenView} from '../../../screen_view'
import stdOutCache from "./cache"
import {LogView} from "../../../components/log_view"

class StdOutView extends ScreenView {
    elem: HTMLDivElement
    uuid: string
    run: Run
    status: Status
    statusCache: RunStatusCache
    runCache: RunCache
    actualWidth: number
    outputContainer: HTMLDivElement
    logView: LogView
    runHeaderCard: RunHeaderCard
    filter: Filter
    private loader: DataLoader
    private refresh: AwesomeRefreshButton
    private stdOut: Logs

    constructor(uuid: string) {
        super()

        this.uuid = uuid
        this.runCache = CACHE.getRun(this.uuid)
        this.statusCache = CACHE.getRunStatus(this.uuid)
        this.filter = new Filter({})

        this.loader = new DataLoader(async (force) => {
            this.status = await this.statusCache.get(force)
            this.stdOut = await stdOutCache.getLogCache(this.uuid).getLast(force)
            this.run = await this.runCache.get(force)
        })
        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))
        this.logView = new LogView(new Logs(<LogModel>{pages: {}, page_length: 0}), async (currentPage): Promise<Logs> => {
            return await stdOutCache.getLogCache(this.uuid).getPage(currentPage, false)
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
        setTitle({section: 'Standard Out'})
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
                    $('h2', '.header.text-center', 'Standard Out')
                    this.loader.render($)
                    this.outputContainer = $('div', '.terminal-card' , $ => {
                        this.logView.render($)
                    })
                })
            })
        })

        try {
            await this.loader.load()

            setTitle({section: 'Standard Out', item: this.run.name})
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
        this.logView.addLogs(this.stdOut)
    }
}

export class StdOutHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('run/:uuid/stdout', [this.handleStdOut])
    }

    handleStdOut = (uuid: string) => {
        SCREEN.setView(new StdOutView(uuid))
    }
}
