import {ROUTER, SCREEN} from '../app'
import {Weya as $, WeyaElement} from '../../../lib/weya/weya'
import {DataLoader} from "../components/loader"
import CACHE, {SessionsListCache} from "../cache/cache"
import {SearchView} from '../components/search';
import {CancelButton, DeleteButton, EditButton} from '../components/buttons'
import {SessionsListItemModel} from '../models/session_list'
import {SessionsListItemView} from '../components/sessions_list_item'
import {HamburgerMenuView} from '../components/hamburger_menu'
import EmptySessionsList from "./empty_sessions_list"
import {UserMessages} from "../components/user_messages"
import {AwesomeRefreshButton} from '../components/refresh_button'
import {handleNetworkErrorInplace} from '../utils/redirect'
import {setTitle} from '../utils/document'
import {ScreenView} from '../screen_view'

class SessionsListView extends ScreenView {
    sessionListCache: SessionsListCache
    currentSessionsList: SessionsListItemModel[]
    elem: HTMLDivElement
    sessionsListContainer: HTMLDivElement
    searchQuery: string
    buttonContainer: HTMLDivElement
    alertContainer: HTMLDivElement
    deleteButton: DeleteButton
    editButton: EditButton
    cancelButton: CancelButton
    isEditMode: boolean
    sessionsDeleteSet: Set<string>
    private loader: DataLoader
    private userMessages: UserMessages
    private refresh: AwesomeRefreshButton

    constructor() {
        super()

        this.sessionListCache = CACHE.getSessionsList()

        this.deleteButton = new DeleteButton({onButtonClick: this.onDelete, parent: this.constructor.name})
        this.editButton = new EditButton({onButtonClick: this.onEdit, parent: this.constructor.name})
        this.cancelButton = new CancelButton({onButtonClick: this.onCancel, parent: this.constructor.name})

        this.userMessages = new UserMessages()

        this.loader = new DataLoader(async (force) => {
            this.currentSessionsList = (await this.sessionListCache.get(force)).sessions
        })
        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))

        this.searchQuery = ''
        this.isEditMode = false
        this.sessionsDeleteSet = new Set<string>()

    }

    async _render() {
        setTitle({section: 'Computers'})
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', $ => {
                this.userMessages.render($)
                new HamburgerMenuView({
                    title: 'Computers',
                    setButtonContainer: container => this.buttonContainer = container
                }).render($)

                $('div', '.runs-list', $ => {
                    new SearchView({onSearch: this.onSearch}).render($)
                    this.loader.render($)
                    this.sessionsListContainer = $('div', '.list.runs-list.list-group', '')
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
        let noRuns = this.currentSessionsList.length == 0
        this.deleteButton.hide(noRuns || !this.isEditMode)
        this.cancelButton.hide(noRuns || !this.isEditMode)
        this.editButton.hide(noRuns || this.isEditMode)
        if (!noRuns && !this.isEditMode) {
            this.refresh.start()
        } else {
            this.refresh.stop()
        }
    }

    sessionsFilter = (session: SessionsListItemModel, query: RegExp) => {
        let name = session.name.toLowerCase()
        let comment = session.comment.toLowerCase()

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
        this.isEditMode = true
        this.refresh.disabled = true
        this.deleteButton.disabled = this.sessionsDeleteSet.size === 0
        this.updateButtons()
    }

    onDelete = async () => {
        try {
            await this.sessionListCache.deleteSessions(this.sessionsDeleteSet)

            this.isEditMode = false
            this.sessionsDeleteSet.clear()
            this.deleteButton.disabled = this.sessionsDeleteSet.size === 0

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
        this.sessionsDeleteSet.clear()
        this.renderList().then()
    }

    onItemClicked = (elem: SessionsListItemView) => {
        let uuid = elem.item.session_uuid
        if (!this.isEditMode) {
            ROUTER.navigate(`/session/${uuid}`)
            return
        }

        if (this.sessionsDeleteSet.has(uuid)) {
            this.sessionsDeleteSet.delete(uuid)
            elem.elem.classList.remove('selected')
        } else {
            this.sessionsDeleteSet.add(uuid)
            elem.elem.classList.add('selected')
        }
        this.deleteButton.disabled = this.sessionsDeleteSet.size === 0
    }

    onSearch = async (query: string) => {
        this.searchQuery = query
        await this.loader.load()
        this.renderList().then()
    }

    private async renderList() {
        if (this.currentSessionsList.length > 0) {
            let re = new RegExp(this.searchQuery.toLowerCase(), 'g')
            this.currentSessionsList = this.currentSessionsList.filter(session => this.sessionsFilter(session, re))

            this.sessionsListContainer.innerHTML = ''
            $(this.sessionsListContainer, $ => {
                for (let i = 0; i < this.currentSessionsList.length; i++) {
                    new SessionsListItemView({
                        item: this.currentSessionsList[i],
                        onClick: this.onItemClicked
                    }).render($)
                }
            })
        } else {
            this.sessionsListContainer.innerHTML = ''
            $(this.sessionsListContainer, $ => {
                new EmptySessionsList().render($)
            })
        }
        this.updateButtons()
    }
}

export class SessionsListHandler {
    constructor() {
        ROUTER.route('computers', [this.handleSessionsList])
    }

    handleSessionsList = () => {
        SCREEN.setView(new SessionsListView())
    }
}
