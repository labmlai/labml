import {ROUTER, SCREEN} from '../app'
import {Weya as $, WeyaElement} from '../../../lib/weya/weya'
import {DataLoader} from "../components/loader"
import CACHE, {RunsFolder, RunsListCache} from "../cache/cache"
import {RunListItem, RunListItemModel} from '../models/run_list'
import {RunsListItemView} from '../components/runs_list_item'
import {SearchView} from '../components/search'
import {CancelButton, DeleteButton, EditButton, IconButton} from '../components/buttons'
import {HamburgerMenuView} from '../components/hamburger_menu'
import EmptyRunsList from './empty_runs_list'
import {UserMessages} from '../components/user_messages'
import {AwesomeRefreshButton} from '../components/refresh_button'
import {handleNetworkErrorInplace} from '../utils/redirect'
import {getQueryParameter, setTitle} from '../utils/document'
import {ScreenView} from '../screen_view'
import {DefaultLineGradient} from "../components/charts/chart_gradients";
import {ErrorResponse} from "../network";

class RunsListView extends ScreenView {
    runListCache: RunsListCache
    runsList: RunListItem[]
    currentRunsList: RunListItem[]
    elem: HTMLDivElement
    runsListContainer: HTMLDivElement
    searchQuery: string
    buttonContainer: HTMLDivElement
    deleteButton: DeleteButton
    archiveButton: IconButton
    editButton: EditButton
    cancelButton: CancelButton
    isEditMode: boolean
    selectedRunsSet: Set<RunListItemModel>
    private loader: DataLoader
    private refresh: AwesomeRefreshButton
    private isTBProcessing: boolean
    private actualWidth: number
    private readonly folder: string

    constructor(folder: string) {
        super()

        this.folder = folder

        this.runListCache = CACHE.getRunsList(this.folder)

        this.deleteButton = new DeleteButton({onButtonClick: this.onDelete, parent: this.constructor.name})
        this.editButton = new EditButton({onButtonClick: this.onEdit, parent: this.constructor.name})
        this.cancelButton = new CancelButton({onButtonClick: this.onCancel, parent: this.constructor.name})
        this.archiveButton = new IconButton({
            onButtonClick: this.onArchiveClick,
            parent: this.constructor.name
        }, folder == RunsFolder.DEFAULT ? '.fas.fa-archive' : '.fas.fa-upload')

        this.loader = new DataLoader(async (force) => {
            let runsList = (await this.runListCache.get(force)).runs
            this.runsList = []
            for (let run of runsList) {
                this.runsList.push(new RunListItem(run))
            }
            this.currentRunsList = this.runsList.slice()
        })
        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))

        this.searchQuery = getQueryParameter('query', window.location.search)
        this.isEditMode = false
        this.selectedRunsSet = new Set<RunListItemModel>()
        this.isTBProcessing = false

    }

    onResize(width: number) {
        super.onResize(width)

        this.actualWidth = Math.min(800, width)

        if (this.elem) {
            this._render().then()
        }
    }

    async _render() {
        setTitle({section: 'Runs'})
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', $ => {
                new HamburgerMenuView({
                    title: (this.folder == RunsFolder.ARCHIVE ? 'Archived ' : '') + 'Runs',
                    setButtonContainer: container => this.buttonContainer = container
                }).render($)

                $('div', '.runs-list', $ => {
                    new SearchView({onSearch: this.onSearch, initText: this.searchQuery}).render($)
                    this.loader.render($)
                    $('svg', {style: {height: `${1}px`}}, $ => {
                        new DefaultLineGradient().render($)
                    })
                    this.runsListContainer = $('div', '.list.runs-list.list-group', '')
                })
            })
        })
        $(this.buttonContainer, $ => {
            this.deleteButton.render($)
            this.archiveButton.render($)
            this.cancelButton.render($)
            this.editButton.render($)
            this.refresh.render($)
            this.deleteButton.hide(true)
            this.archiveButton.hide(true)
            this.cancelButton.hide(true)
            this.editButton.hide(true)
        })

        try {
            await this.loader.load()

            this.renderList()
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
        this.refresh.stop()
    }

    updateButtons() {
        let noRuns = this.currentRunsList.length == 0

        this.deleteButton.hide((noRuns || !this.isEditMode) ||
            (this.folder != RunsFolder.DEFAULT && this.folder != RunsFolder.ARCHIVE))
        this.cancelButton.hide(noRuns || !this.isEditMode)
        this.editButton.hide(noRuns || this.isEditMode)
        this.archiveButton.hide((noRuns || !this.isEditMode) ||
            (this.folder != RunsFolder.DEFAULT && this.folder != RunsFolder.ARCHIVE))

        if (!noRuns && !this.isEditMode) {
            this.refresh.start()
        } else {
            this.refresh.stop()
        }
    }

    runsFilter = (run: RunListItemModel, query: RegExp) => {
        let name = run.name.toLowerCase()
        let comment = run.comment.toLowerCase()
        let tags = run.tags.join(' ').toLowerCase()

        return (name.search(query) !== -1 || comment.search(query) !== -1 || tags.search(query) !== -1)
    }

    onRefresh = async () => {
        this.editButton.disabled = true
        try {
            await this.loader.load(true)

            await this.renderList()
        } catch (e) {

        } finally {
            this.editButton.disabled = false
        }
    }

    onEdit = () => {
        let isRunsSelected = this.selectedRunsSet.size === 0

        this.isEditMode = true
        this.refresh.disabled = true
        this.deleteButton.disabled = isRunsSelected
        this.archiveButton.disabled = isRunsSelected
        this.updateButtons()
    }

    onArchiveClick = async () => {
        try {
            let runUUIDs: Array<string> = []
            for (let runListItem of this.selectedRunsSet) {
                runUUIDs.push(runListItem.run_uuid)
            }

            let response: ErrorResponse
            if (this.folder == RunsFolder.DEFAULT) {
                response = await CACHE.archiveRuns(runUUIDs)
            } else if (this.folder == RunsFolder.ARCHIVE) {
                response = await CACHE.unarchiveRuns(runUUIDs)
            }

            if (response.is_successful == false) {
                UserMessages.shared.error(response.error ?? `Failed to ${
                    this.folder == RunsFolder.DEFAULT ? '': 'Un'}archive runs. is_successful=false from server`)
                return
            }

            this.isEditMode = false
            this.selectedRunsSet.clear()
            this.archiveButton.disabled = this.selectedRunsSet.size === 0

            await this.loader.load()

            this.refresh.disabled = false
        } catch (e) {
            if (this.folder == RunsFolder.DEFAULT)
                UserMessages.shared.networkError(e, 'Failed to archive runs')
            else
                UserMessages.shared.networkError(e, 'Failed to unarchive runs')
            return
        }

        this.renderList()
    }

    onDelete = async () => {
        try {
            let runUUIDs: Array<string> = []
            for (let runListItem of this.selectedRunsSet) {
                runUUIDs.push(runListItem.run_uuid)
            }

            await this.runListCache.deleteRuns(runUUIDs)

            this.isEditMode = false
            this.selectedRunsSet.clear()
            this.deleteButton.disabled = this.selectedRunsSet.size === 0

            await this.loader.load()


        } catch (e) {
            UserMessages.shared.networkError(e, 'Failed to delete runs')
            return
        } finally {
            this.refresh.disabled = false
        }

        this.renderList()
    }

    onCancel = () => {
        this.isEditMode = false
        this.refresh.disabled = false
        this.selectedRunsSet.clear()
        this.renderList()
    }

    onItemClicked = (elem: RunsListItemView) => {
        let runListItem = elem.item

        if (!this.isEditMode) {
            ROUTER.navigate(`/run/${runListItem.run_uuid}`)
            return
        }

        if (this.selectedRunsSet.has(runListItem)) {
            this.selectedRunsSet.delete(runListItem)
            elem.elem.classList.remove('selected')
        } else {
            this.selectedRunsSet.add(runListItem)
            elem.elem.classList.add('selected')
        }

        let isRunsSelected = this.selectedRunsSet.size === 0

        this.deleteButton.disabled = isRunsSelected || this.isTBProcessing
        this.archiveButton.disabled = isRunsSelected
    }

    onSearch = async (query: string) => {
        this.searchQuery = query
        window.history.replaceState({}, "", `${window.location.toString().replace(window.location.search, "")}?query=${encodeURIComponent(query)}`)
        this.renderList()
    }

    private renderList() {
        if (this.runsList.length > 0) {
            let re = new RegExp(this.searchQuery.toLowerCase(), 'g')
            this.currentRunsList = this.runsList.filter(run => this.runsFilter(run, re))

            this.runsListContainer.innerHTML = ''
            $(this.runsListContainer, $ => {
                for (let i = 0; i < this.currentRunsList.length; i++) {
                    new RunsListItemView({
                        item: this.currentRunsList[i],
                        onClick: this.onItemClicked,
                        width: this.actualWidth}).render($)
                }
            })
        } else {
            this.runsListContainer.innerHTML = ''
            $(this.runsListContainer, $ => {
                new EmptyRunsList().render($)
            })
        }
        this.updateButtons()
    }
}

export class RunsListHandler {
    constructor() {
        ROUTER.route('runs', [this.handleRunsList])
        ROUTER.route('runs/:folder', [this.handleFolder])
    }

    handleFolder = (folder: string) => {
        SCREEN.setView(new RunsListView(folder))
    }

    handleRunsList = () => {
        SCREEN.setView(new RunsListView(RunsFolder.DEFAULT))
    }
}
