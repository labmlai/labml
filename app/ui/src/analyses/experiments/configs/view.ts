import {ROUTER, SCREEN} from "../../../app"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {Run} from "../../../models/run"
import {Status} from "../../../models/status"
import CACHE, {RunCache, RunStatusCache} from "../../../cache/cache"
import {DataLoader} from "../../../components/loader"
import {BackButton, SaveButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import {Configs, ConfigStatus} from "./components"
import {ViewHandler} from "../../types"
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {setTitle} from '../../../utils/document'
import {ScreenView} from '../../../screen_view'
import {UserMessages} from "../../../components/user_messages"

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
    private save: SaveButton
    private configsChanged: boolean
    private userMessage: UserMessages

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
        this.save = new SaveButton({onButtonClick: this.onSave.bind(this), parent: this.constructor.name, isDisabled: true})
        this.configsChanged = false
        this.userMessage = new UserMessages()
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
                this.userMessage.render($)
                    $('div', $ => {
                        $('div', '.nav-container', $ => {
                            new BackButton({text: 'Run', parent: this.constructor.name}).render($)
                            this.save.render($)
                            this.refresh.render($)
                        })
                        this.runHeaderCard = new RunHeaderCard({
                            uuid: this.uuid,
                            width: this.actualWidth
                        })
                        this.runHeaderCard.render($).then()
                        $('h2', '.header.text-center', 'Configurations')
                        this.loader.render($)
                        this.configsContainer = $('div')
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

    async onSave() {
        if (!this.configsChanged) {
            return
        }
        try {
            this.save.disabled = true
            this.save.loading = true
            this.configsChanged = false

            let data = {
                'favourite_configs': this.run.favourite_configs,
                'selected_configs': this.run.selected_configs
            }

            await CACHE.getRun(this.uuid).updateRunData(data)
            await CACHE.getRunsList(this.run.folder).localUpdateRun(this.run)
        } catch (e) {
            this.userMessage.networkError(e, "Failed to save configurations")
            this.save.disabled = false
            this.configsChanged = true
        } finally {
            this.save.loading = false
        }
    }

    async onRefresh() {
        try {
            await this.loader.load(true)
        } catch (e) {
            this.userMessage.networkError(e, "Refresh failed")
            return
        } finally {
            if (this.status && !this.status.isRunning) {
                this.refresh.stop()
            }
            await this.runHeaderCard.refresh().then()
        }

        this.renderConfigsView()
    }

    onVisibilityChange() {
        this.refresh.changeVisibility(!document.hidden)
    }

    protected onTap(key: string, configStatus: ConfigStatus) {
        this.configsChanged = true
        this.save.disabled = false

        if (this.run.selected_configs == null) {
            this.run.selected_configs = []
        } else if (this.run.favourite_configs == null) {
            this.run.favourite_configs = []
        }

        let selectedIndex: number = this.run.selected_configs.indexOf(key)
        let isSelected: boolean = selectedIndex >= 0
        let favouriteIndex: number = this.run.favourite_configs.indexOf(key)
        let isFavourite: boolean = favouriteIndex >= 0

        // cleanup before updates
        if (isSelected) {
            this.run.selected_configs.splice(selectedIndex, 1)
        }
        if (isFavourite) {
            this.run.favourite_configs.splice(favouriteIndex, 1)
        }

        if (configStatus == ConfigStatus.SELECTED || configStatus == ConfigStatus.FAVOURITE) {
            this.run.selected_configs.push(key)
        }
        if (configStatus == ConfigStatus.FAVOURITE) {
            this.run.favourite_configs.push(key)
        }

        this.run.updateConfigs()
    }

    renderConfigsView() {
        this.configsContainer.innerHTML = ''
        $(this.configsContainer, $ => {
            new Configs({configs: this.run.configs, width: this.actualWidth, isSummary: false, onTap: this.onTap.bind(this)}).render($)
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
