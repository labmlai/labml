import {ROUTER, SCREEN} from "../../../app"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {Run} from "../../../models/run"
import {Status} from "../../../models/status"
import CACHE, {RunCache, RunStatusCache} from "../../../cache/cache"
import {DataLoader} from "../../../components/loader"
import {BackButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import {Configs} from "./components"
import mix_panel from "../../../mix_panel"
import {ViewHandler} from "../../types"
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {setTitle} from '../../../utils/document'
import {ScreenView} from '../../../screen_view'

class RunConfigsView extends ScreenView {
    elem: HTMLDivElement
    uuid: string
    run: Run
    status: Status
    statusCache: RunStatusCache
    runCache: RunCache
    actualWidth: number
    runHeaderCard: RunHeaderCard
    configsContainer: HTMLDivElement
    private loader: DataLoader
    private refresh: AwesomeRefreshButton

    constructor(uuid: string) {
        super()

        this.uuid = uuid
        this.runCache = CACHE.getRun(this.uuid)
        this.statusCache = CACHE.getRunStatus(this.uuid)

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
        setTitle({section: 'Configurations'})
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', '.page',
                {style: {width: `${this.actualWidth}px`}}, $ => {
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
                        $('h2', '.header.text-center', 'Configurations')
                        this.loader.render($)
                        this.configsContainer = $('div', '.labml-card')
                    })
                })
        })

        try {
            await this.loader.load()

            setTitle({section: 'Configurations', item: this.run.name})
            this.renderConfigsView()
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
            this.renderConfigsView()
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

    renderConfigsView() {
        this.configsContainer.innerHTML = ''
        $(this.configsContainer, $ => {
            new Configs({configs: this.run.configs, width: this.actualWidth, isHyperParamOnly: false}).render($)
        })
    }

}

export class RunConfigsHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('run/:uuid/configs', [this.handleRunConfigs])
    }

    handleRunConfigs = (uuid: string) => {
        SCREEN.setView(new RunConfigsView(uuid))
    }
}
