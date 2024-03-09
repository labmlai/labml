import {ROUTER, SCREEN} from '../app'
import {Weya as $, WeyaElement} from '../../../lib/weya/weya'
import {DataLoader} from "../components/loader"
import CACHE, {RunsListCache} from "../cache/cache"
import {RunListItem, RunListItemModel} from '../models/run_list'
import {RunsListItemView} from '../components/runs_list_item'
import {SearchView} from '../components/search'
import {CancelButton, DeleteButton, EditButton} from '../components/buttons'
import {HamburgerMenuView} from '../components/hamburger_menu'
import mix_panel from "../mix_panel"
import EmptyRunsList from './empty_runs_list'
import {UserMessages} from '../components/user_messages'
import {AwesomeRefreshButton} from '../components/refresh_button'
import {handleNetworkErrorInplace} from '../utils/redirect'
import {setTitle} from '../utils/document'
import {ScreenView} from '../screen_view'
import {DefaultLineGradient} from "../components/charts/chart_gradients";

class RunsListView extends ScreenView {
    runListCache: RunsListCache
    currentRunsList: RunListItem[]
    elem: HTMLDivElement
    runsListContainer: HTMLDivElement
    searchQuery: string
    buttonContainer: HTMLDivElement
    deleteButton: DeleteButton
    editButton: EditButton
    cancelButton: CancelButton
    isEditMode: boolean
    selectedRunsSet: Set<RunListItemModel>
    private loader: DataLoader
    private userMessages: UserMessages
    private refresh: AwesomeRefreshButton
    private isTBProcessing: boolean
    private actualWidth: number

    constructor() {
        super()

        this.runListCache = CACHE.getRunsList()

        this.deleteButton = new DeleteButton({onButtonClick: this.onDelete, parent: this.constructor.name})
        this.editButton = new EditButton({onButtonClick: this.onEdit, parent: this.constructor.name})
        this.cancelButton = new CancelButton({onButtonClick: this.onCancel, parent: this.constructor.name})

        this.userMessages = new UserMessages()

        this.loader = new DataLoader(async (force) => {
            let runsList = (await this.runListCache.get(force)).runs
            this.currentRunsList = []
            for (let run of runsList) {
                this.currentRunsList.push(new RunListItem(run))
            }
        })
        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))

        this.searchQuery = ''
        this.isEditMode = false
        this.selectedRunsSet = new Set<RunListItemModel>()
        this.isTBProcessing = false

        mix_panel.track('Runs List View')
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
                this.userMessages.render($)
                new HamburgerMenuView({
                    title: 'Runs',
                    setButtonContainer: container => this.buttonContainer = container
                }).render($)

                $('div', '.runs-list', $ => {
                    new SearchView({onSearch: this.onSearch}).render($)
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
            this.cancelButton.render($)
            this.editButton.render($)
            this.refresh.render($)
            this.deleteButton.hide(true)
            this.cancelButton.hide(true)
            this.editButton.hide(true)
        })

        try {
            await this.loader.load()

            this.renderList().then()
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

        this.deleteButton.hide(noRuns || !this.isEditMode)
        this.cancelButton.hide(noRuns || !this.isEditMode)
        this.editButton.hide(noRuns || this.isEditMode)

        if (!noRuns && !this.isEditMode) {
            this.refresh.start()
        } else {
            this.refresh.stop()
        }
    }

    runsFilter = (run: RunListItemModel, query: RegExp) => {
        let name = run.name.toLowerCase()
        let comment = run.comment.toLowerCase()

        return (name.search(query) !== -1 || comment.search(query) !== -1)
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
        this.updateButtons()
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
            await this.renderList()
            this.refresh.disabled = false
        } catch (e) {
            this.userMessages.networkError()
        }
    }

    onCancel = () => {
        this.isEditMode = false
        this.refresh.disabled = false
        this.selectedRunsSet.clear()
        this.renderList().then()
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
    }

    onSearch = async (query: string) => {
        this.searchQuery = query
        await this.loader.load()
        this.renderList().then()
    }

    private async renderList() {
        if (this.currentRunsList.length > 0) {
            let re = new RegExp(this.searchQuery.toLowerCase(), 'g')
            this.currentRunsList = this.currentRunsList.filter(run => this.runsFilter(run, re))

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
    }

    handleRunsList = () => {
        SCREEN.setView(new RunsListView())
    }
}
