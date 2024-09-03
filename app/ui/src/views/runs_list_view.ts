import {ROUTER, SCREEN} from '../app'
import {Weya as $, WeyaElement} from '../../../lib/weya/weya'
import {DataLoader} from "../components/loader"
import CACHE, {RunsListCache} from "../cache/cache"
import {RunListItem, RunListItemModel} from '../models/run_list'
import {RunsListItemView} from '../components/runs_list_item'
import {SearchView} from '../components/search'
import {CancelButton, DeleteButton, EditButton} from '../components/buttons'
import {HamburgerMenuView} from '../components/hamburger_menu'
import EmptyRunsList from './empty_runs_list'
import {UserMessages} from '../components/user_messages'
import {AwesomeRefreshButton} from '../components/refresh_button'
import {handleNetworkErrorInplace} from '../utils/redirect'
import {getQueryParameter, setTitle} from '../utils/document'
import {ScreenView} from '../screen_view'
import {DefaultLineGradient} from "../components/charts/chart_gradients"
import {extractTags, getSearchQuery, runsFilter} from "../utils/search"

class RunsListView extends ScreenView {
    runListCache: RunsListCache
    runsList: RunListItem[]
    currentRunsList: RunListItem[]
    elem: HTMLDivElement
    runsListContainer: HTMLDivElement
    searchQuery: string
    searchView: SearchView
    buttonContainer: HTMLDivElement
    deleteButton: DeleteButton
    editButton: EditButton
    cancelButton: CancelButton
    isEditMode: boolean
    selectedRunsSet: Set<RunListItemModel>
    private loader: DataLoader
    private refresh: AwesomeRefreshButton
    private isTBProcessing: boolean
    private actualWidth: number
    private defaultTag: string // permanent tag in the url

    constructor(tag: string) {
        super()

        this.defaultTag = tag

        this.runListCache = CACHE.getRunsList()

        this.deleteButton = new DeleteButton({onButtonClick: this.onDelete, parent: this.constructor.name})
        this.editButton = new EditButton({onButtonClick: this.onEdit, parent: this.constructor.name})
        this.cancelButton = new CancelButton({onButtonClick: this.onCancel, parent: this.constructor.name})


        this.loader = new DataLoader(async (force) => {
            let runsList = (await this.runListCache.get(force, this.defaultTag)).runs
            this.runsList = []
            for (let run of runsList) {
                this.runsList.push(new RunListItem(run))
            }
            this.currentRunsList = this.runsList.slice()
        })
        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))
        this.searchView = new SearchView({onSearch: this.onSearch, initText: this.searchQuery})

        this.searchQuery = getQueryParameter('query', window.location.search)
        let tags = getQueryParameter('tags', window.location.search)

        if (this.defaultTag) {
            this.searchQuery += ` $${this.defaultTag}`
        }
        if (tags) {
            for (let tag of tags.split(',')) {
                this.searchQuery += ` :${tag}`
            }
        }

        if (this.searchQuery === "") {
            this.searchQuery = getSearchQuery()
            let r = extractTags(this.searchQuery)
            this.defaultTag = r.mainTags.length > 0 ? r.mainTags[0] : ""
        }

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
                    title: 'Runs',
                    setButtonContainer: container => this.buttonContainer = container
                }).render($)

                $('div', '.runs-list', $ => {
                    this.searchView.render($)
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

        this.deleteButton.hide((noRuns || !this.isEditMode))
        this.cancelButton.hide(noRuns || !this.isEditMode)
        this.editButton.hide(noRuns || this.isEditMode)

        if (!noRuns && !this.isEditMode) {
            this.refresh.start()
        } else {
            this.refresh.stop()
        }
    }

    onRefresh = async () => {
        this.editButton.disabled = true
        try {
            await this.loader.load(true)

            this.renderList()
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
    }

    onSearch = async (query: string) => {
        this.searchView.hideLoader(false)
        this.searchQuery = query
        let r = extractTags(query)

        let mainTag = r.mainTags.length > 0 ? r.mainTags[0] : ""
        let tags = r.tags.concat(r.mainTags).filter(tag => tag !== mainTag)

        let queryString = (r.query == "" ? "" : `query=${encodeURIComponent(r.query)}`)
        let tagsString = (tags.length == 0 ? "" : `tags=${encodeURIComponent(tags.join(','))}`)
        if (queryString && tagsString) {
            queryString += "&"
        }
        window.history.replaceState({}, "", `/runs${mainTag ? "/"+mainTag : ""}${queryString || tagsString ? "?" : ""}${queryString}${tagsString}`)

        this.defaultTag = mainTag
        await this.loader.load()

        this.renderList()
        this.searchView.hideLoader(true)
    }

    private renderList() {
        if (this.runsList.length > 0) {
            this.currentRunsList = this.runsList.filter(run => runsFilter(run, this.searchQuery))

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
        ROUTER.route('runs/:tag', [this.handleTag])
    }

    handleTag = (tag: string) => {
        SCREEN.setView(new RunsListView(tag))
    }

    handleRunsList = () => {
        SCREEN.setView(new RunsListView(""))
    }
}
