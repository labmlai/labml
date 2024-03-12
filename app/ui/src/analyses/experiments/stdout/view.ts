import {Run} from "../../../models/run"
import CACHE, {RunCache, RunStatusCache} from "../../../cache/cache"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import Filter from "../../../utils/ansi_to_html"
import {Status} from "../../../models/status"
import {ROUTER, SCREEN} from "../../../app"
import {BackButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import {DataLoader} from "../../../components/loader"
import mix_panel from "../../../mix_panel"
import {ViewHandler} from "../../types"
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {setTitle} from '../../../utils/document'
import {ScreenView} from '../../../screen_view'

class StdOutView extends ScreenView {
    elem: HTMLDivElement
    uuid: string
    run: Run
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
                    this.outputContainer = $('div', '.terminal-card')
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
        // escape to avoid rendering html like chars
        this.run.stdout = this.run.stdout.replace(/</g, '&lt;');
        this.run.stdout = this.run.stdout.replace(/>/g, '&gt;');
        // this.run.stdout = this.run.stdout.replace(/&/g, '&amp;');

        this.outputContainer.innerHTML = ''
        $(this.outputContainer, $ => {
            let output = $('pre', '')
            output.innerHTML = this.filter.toHtml(this.run.stdout)
        })
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
