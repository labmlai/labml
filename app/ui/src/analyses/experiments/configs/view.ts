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
import {SearchView} from "../../../components/search"
import {Config} from "../../../models/config"

const CONFIG_ATTRIBUTES = ['meta', 'custom', 'only_option', 'explicit', 'explicitly_specified', 'hp', 'hyperparam']

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
    private searchQuery: string
    private searchView: SearchView
    private loader: DataLoader
    private refresh: AwesomeRefreshButton
    private save: SaveButton
    private configsChanged: boolean
    private currentConfigs: Config[]

    constructor(uuid: string) {
        super()

        this.uuid = uuid
        this.runCache = CACHE.getRun(this.uuid)
        this.statusCache = CACHE.getRunStatus(this.uuid)

        this.loader = new DataLoader(async (force) => {
            this.status = await this.statusCache.get(force)
            this.run = await this.runCache.get(force)
            this.currentConfigs = this.run.configs
        })
        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))
        this.save = new SaveButton({
            onButtonClick: this.onSave.bind(this),
            parent: this.constructor.name,
            isDisabled: true
        })
        this.searchQuery = 'is:explicit'
        this.searchView = new SearchView({
            onSearch: this.renderSearchConfigView.bind(this),
            initText: this.searchQuery
        })
        this.configsChanged = false
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

    private renderSearchConfigView(query: string) {
        this.searchQuery = query

        let searchArray = query.split(" ")
        let attributes = []
        let queries = []
        for (let search of searchArray) {
            if (search.includes('is:')) {
                let attribute = search.split(/[: ]+/).pop().toLowerCase()
                if (CONFIG_ATTRIBUTES.includes(attribute)) {
                    attributes.push(attribute)
                } else {
                    queries.push(search)
                }
            } else {
                queries.push(search)
            }
        }

        if (this.run.configs.length > 0) {
            this.currentConfigs = this.run.configs.filter(config => this.configsFilter(config, attributes, queries.join(" ")))
        }
        this.renderConfigsView()
    }

    private configsFilter(config: Config, attributes: string[], query: string) {
        const queryRegex = new RegExp(query.toLowerCase(), 'g')

        let matchName = query == "" || config.name.toLowerCase().search(queryRegex) !== -1
        let matchKey = query == "" || config.key.toLowerCase().search(queryRegex) !== -1
        let hasAttributesInConfig = this.hasAttributesInConfig(config, attributes)

        if (attributes.length > 0) {
            return (matchName || matchKey) && hasAttributesInConfig
        }

        return matchName || matchKey
    }

    private hasAttributesInConfig(config: Config, attributes: string[]) {
        for (let attribute of attributes) {
            if (attribute == 'meta' && config.isMeta) {
                return true
            }
            if (attribute == 'custom' && config.isCustom) {
                return true
            }
            if (attribute == 'only_option' && config.isOnlyOption) {
                return true
            }
            if (attribute == 'explicit' && config.isExplicitlySpecified) {
                return true
            }
            if (attribute == 'explicitly_specified' && config.isExplicitlySpecified) {
                return true
            }
            if (attribute == 'hp' && config.isHyperparam) {
                return true
            }
            if (attribute == 'hyperparam' && config.isHyperparam) {
                return true
            }
        }

        return false
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
                            this.save.render($)
                            this.refresh.render($)
                        })
                        this.runHeaderCard = new RunHeaderCard({
                            uuid: this.uuid,
                            width: this.actualWidth
                        })
                        this.runHeaderCard.render($).then()
                        $('h2', '.header.text-center', 'Configurations')
                        this.searchView.render($)
                        this.loader.render($)
                        this.configsContainer = $('div')
                    })
                })
        })

        try {
            await this.loader.load()

            setTitle({section: 'Configurations', item: this.run.name})
            this.renderSearchConfigView(this.searchQuery)
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
            await CACHE.getRunsList().localUpdateRun(this.run)
        } catch (e) {
            UserMessages.shared.networkError(e, "Failed to save configurations")
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
            UserMessages.shared.networkError(e, "Refresh failed")
            return
        } finally {
            if (this.status && !this.status.isRunning) {
                this.refresh.stop()
            }
            await this.runHeaderCard.refresh().then()
        }

        this.renderSearchConfigView(this.searchQuery)
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
            new Configs({
                configs: this.currentConfigs,
                width: this.actualWidth,
                isSummary: false,
                onTap: this.onTap.bind(this)
            }).render($)
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
