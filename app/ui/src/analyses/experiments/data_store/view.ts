import {ROUTER, SCREEN} from "../../../app"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import CACHE, {DataStoreCache, RunStatusCache} from "../../../cache/cache"
import {DataLoader} from "../../../components/loader"
import {BackButton, CancelButton, EditButton, SaveButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import {ViewHandler} from "../../types"
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {setTitle} from '../../../utils/document'
import {ScreenView} from '../../../screen_view'
import {UserMessages} from "../../../components/user_messages"
import {SearchView} from "../../../components/search"
import {DataStore} from "../../../models/data_store"
import {Status} from "../../../models/status"
import EditableField from "../../../components/input/editable_field"
import {DataStoreComponent} from "../../../components/data_store";

class DataStoreView extends ScreenView {
    private elem: HTMLDivElement
    private readonly uuid: string
    private dataStore: DataStore
    private readonly dataStoreCache: DataStoreCache
    actualWidth: number
    private runHeaderCard: RunHeaderCard
    dataStoreContainer: HTMLDivElement
    private loader: DataLoader
    private refresh: AwesomeRefreshButton
    private save: SaveButton
    private edit: EditButton
    private searchQuery: string
    private searchView: SearchView
    private isEditing: boolean
    private status: Status
    private statusCache: RunStatusCache
    private buttonContainer: HTMLElement
    private dataStoreField: EditableField
    private cancelButton: CancelButton

    constructor(uuid: string) {
        super()

        this.uuid = uuid
        this.dataStoreCache = CACHE.getDataStore(this.uuid)
        this.statusCache = CACHE.getRunStatus(this.uuid)

        this.loader = new DataLoader(async (force) => {
            this.status = await this.statusCache.get(force)
            this.dataStore = await this.dataStoreCache.get(force)

            console.log(this.dataStore.yamlString)
        })
        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))
        this.save = new SaveButton({
            onButtonClick: this.onSave.bind(this),
            parent: this.constructor.name,
            isDisabled: false
        })
        this.searchQuery = ''
        this.searchView = new SearchView({
            onSearch: this.renderDataStore.bind(this),
            initText: this.searchQuery
        })
        this.isEditing = false
        this.edit = new EditButton({
            onButtonClick: this.onEdit,
            isDisabled: false,
            parent:
            this.constructor.name
        })
        this.cancelButton = new CancelButton({
            onButtonClick: this.onEdit,
            parent: this.constructor.name
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
        setTitle({section: 'Data Store'})
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', '.page.data-store',
                {style: {width: `${this.actualWidth}px`}}, $ => {
                    $('div', $ => {
                        $('div', '.nav-container', $ => {
                            new BackButton({text: 'Run', parent: this.constructor.name}).render($)

                            this.buttonContainer = $('span')

                            this.refresh.render($)
                        })
                        this.runHeaderCard = new RunHeaderCard({
                            uuid: this.uuid,
                            width: this.actualWidth
                        })
                        this.runHeaderCard.render($).then()
                        $('h2', '.header.text-center', 'Data Store')
                        this.searchView.render($)
                        this.loader.render($)
                        this.dataStoreContainer = $('div', '')
                    })
                })
        })

        try {
            await this.loader.load()

            setTitle({section: 'Data Store', item: ''})
            this.renderButtons()
            this.renderDataStore(this.searchQuery)
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

    private renderButtons() {
        this.buttonContainer.innerHTML = ''

        if (this.isEditing) {
            $(this.buttonContainer, $ => {
                this.save.render($)
                this.cancelButton.render($)
            })
        } else {
            $(this.buttonContainer, $ => {
                this.edit.render($)
            })
        }
    }

    private renderDataStore(query: string = '') {
        this.searchQuery = query

        this.dataStoreContainer.innerHTML = ''
        $(this.dataStoreContainer, $ => {
            if (this.isEditing) {
                this.dataStoreField = new EditableField({
                    name: "", value: this.dataStore.yamlString,
                    numEditRows: 20,
                    isEditable: true
                })
                this.dataStoreField.render($)
            } else {
                new DataStoreComponent(this.dataStore).render($)
            }
        })
    }

    private onEdit = () => {
        this.isEditing = !this.isEditing

        if (this.isEditing) {
            this.refresh.pause()
        } else {
            this.refresh.resume()
        }


        this.renderButtons()
        this.renderDataStore(this.searchQuery)
    }

    async onSave() {
        this.isEditing = !this.isEditing

        try {
            this.dataStore = await this.dataStoreCache.update(this.dataStoreField.getInput())
        } catch (e) {
            UserMessages.shared.networkError(e, "Failed to save data store")
            return
        } finally {
            if (this.status && this.status.isRunning) {
                this.refresh.resume()
            }

            this.renderButtons()
            this.renderDataStore(this.searchQuery)
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

        this.renderDataStore(this.searchQuery)
    }

    onVisibilityChange() {
        this.refresh.changeVisibility(!document.hidden)
    }
}

export class DataStoreHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('run/:uuid/data_store', [this.handle])
    }

    handle = (uuid: string) => {
        SCREEN.setView(new DataStoreView(uuid))
    }
}
