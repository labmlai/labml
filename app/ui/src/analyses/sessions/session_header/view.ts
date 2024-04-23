import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {ROUTER, SCREEN} from "../../../app"
import {Run} from "../../../models/run"
import CACHE, {SessionCache, SessionsListCache, SessionStatusCache, UserCache} from "../../../cache/cache"
import {Status} from "../../../models/status"
import {BackButton, CancelButton, DeleteButton, EditButton, SaveButton} from "../../../components/buttons"
import EditableField from "../../../components/input/editable_field"
import {formatTime, getTimeDiff} from "../../../utils/time"
import {DataLoader} from "../../../components/loader"
import {StatusView} from "../../../components/status"
import {handleNetworkError, handleNetworkErrorInplace} from '../../../utils/redirect'
import {Session} from "../../../models/session"
import {getPath, setTitle} from '../../../utils/document'
import {SessionsListItemModel} from "../../../models/session_list"
import {SessionsListItemView} from "../../../components/sessions_list_item"
import {ScreenView} from '../../../screen_view'
import {User} from '../../../models/user'
import {UserMessages} from "../../../components/user_messages"

class SessionHeaderView extends ScreenView {
    elem: HTMLDivElement
    session: Session
    sessionCache: SessionCache
    sessionsList: SessionsListItemModel[]
    sessionListCache: SessionsListCache
    sessionsListContainer: HTMLDivElement
    status: Status
    statusCache: SessionStatusCache
    user: User
    userCache: UserCache
    isEditMode: boolean
    uuid: string
    actualWidth: number
    nameField: EditableField
    commentField: EditableField
    private fieldContainer: HTMLDivElement
    private deleteButton: DeleteButton
    private loader: DataLoader
    private userMessages: UserMessages

    constructor(uuid: string) {
        super()
        this.uuid = uuid
        this.sessionCache = CACHE.getSession(this.uuid)
        this.statusCache = CACHE.getSessionStatus(this.uuid)
        this.userCache = CACHE.getUser()
        this.sessionListCache = CACHE.getSessionsList()
        this.isEditMode = false

        this.deleteButton = new DeleteButton({onButtonClick: this.onDelete.bind(this), parent: this.constructor.name})

        this.loader = new DataLoader(async (force) => {
            this.status = await this.statusCache.get(force)
            this.session = await this.sessionCache.get(force)
            this.user = await this.userCache.get(force)

            if (this.user.is_complete) {
                let sessionsList = (await this.sessionListCache.get(force)).sessions
                this.sessionsList = sessionsList.filter(session => this.sessionsFilter(session))
            }
        })
        this.userMessages = new UserMessages()
    }

    get requiresAuth(): boolean {
        return false
    }

    sessionsFilter = (session: SessionsListItemModel) => {
        return session.computer_uuid === this.session.computer_uuid && session.session_uuid !== this.session.session_uuid
    }

    onResize(width: number) {
        super.onResize(width)

        this.actualWidth = Math.min(800, width)

        if (this.elem) {
            this._render().then()
        }
    }

    async _render() {
        setTitle({section: 'Computer Details'})
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', '.page',
                {style: {width: `${this.actualWidth}px`}},
                $ => {
                    $('div', $ => {
                        this.userMessages.render($)
                        $('div', '.nav-container', $ => {
                            new BackButton({text: 'Run', parent: this.constructor.name}).render($)
                            if (this.isEditMode) {
                                new CancelButton({
                                    onButtonClick: this.onToggleEdit,
                                    parent: this.constructor.name
                                }).render($)
                                new SaveButton({onButtonClick: this.updateRun, parent: this.constructor.name}).render($)
                                this.deleteButton.render($)
                                this.deleteButton.hide(true)
                            } else {
                                new EditButton({
                                    onButtonClick: this.onToggleEdit,
                                    parent: this.constructor.name
                                }).render($)
                            }
                        })
                        $('h2', '.header.text-center', 'Computer Details')
                        this.loader.render($)
                        this.fieldContainer = $('div', '.input-list-container')
                    })

                    $('h6', '.text-center', 'More Sessions')
                    $('div', '.runs-list', $ => {
                        this.sessionsListContainer = $('div', '.list.runs-list.list-group', '')
                    })
                })
        })

        try {
            await this.loader.load()

            setTitle({section: 'Computer Details', item: this.session.name})
            this.renderFields()
            this.renderSessionsList()
        } catch (e) {
            handleNetworkErrorInplace(e)
        } finally {

        }
    }

    render(): WeyaElement {
        this.elem = $('div')

        this._render().then()

        return this.elem
    }

    renderFields() {
        this.fieldContainer.innerHTML = ''
        $(this.fieldContainer, $ => {
            $('ul', $ => {
                this.nameField = new EditableField({
                    name: 'Run Name',
                    value: this.session.name,
                    isEditable: this.isEditMode
                })
                this.nameField.render($)
                this.commentField = new EditableField({
                    name: 'Comment',
                    value: this.session.comment,
                    isEditable: this.isEditMode
                })
                this.commentField.render($)
                $(`li`, $ => {
                    $('span.item-key', 'Run Status')
                    $('span.item-value', $ => {
                        new StatusView({status: this.status.run_status, type: 'session'}).render($)
                    })
                })
                new EditableField({
                    name: 'Computer UUID',
                    value: this.session.computer_uuid,
                }).render($)
                new EditableField({
                    name: 'Session UUID',
                    value: this.session.session_uuid,
                }).render($)
                new EditableField({
                    name: 'Start Time',
                    value: formatTime(this.session.start_time),
                }).render($)
                new EditableField({
                    name: 'Last Recorded',
                    value: this.status.isRunning ? getTimeDiff(this.status.last_updated_time * 1000) :
                        formatTime(this.status.last_updated_time),
                }).render($)
            })
        })
        this.deleteButton.hide(!(this.user.is_complete && this.session.is_claimed))
    }

    onItemClicked = (elem: SessionsListItemView) => {
        let uuid = elem.item.session_uuid
        if (!this.isEditMode) {
            ROUTER.navigate(`/session/${uuid}`)
            return
        }
    }

    renderSessionsList() {
        this.sessionsListContainer.innerHTML = ''
        $(this.sessionsListContainer, $ => {
            if (!this.user.is_complete) {
                $('div', '.text-center', $ => {
                    $('span', 'You need to be authenticated to view this content. ')
                    let linkElem = $('a', '.generic-link', {href: '/auth/sign_in', on: {click: this.handleSignIn}})
                    linkElem.textContent = 'Sign In'
                })

                return
            }
            for (let i = 0; i < this.sessionsList.length; i++) {
                new SessionsListItemView({
                    item: this.sessionsList[i],
                    onClick: this.onItemClicked
                }).render($)
            }
        })
    }

    onToggleEdit = () => {
        this.isEditMode = !this.isEditMode

        this._render().then()
    }

    onDelete = async () => {
        if (confirm("Are you sure?")) {
            try {
                await CACHE.getSessionsList().deleteSessions(new Set<string>([this.uuid]))
                ROUTER.navigate('/computers')
            } catch (e) {
                this.userMessages.networkError(e, "Failed to delete session")
                return
            }
        }
    }

    updateRun = () => {
        if (this.nameField.getInput()) {
            this.session.name = this.nameField.getInput()
        }

        if (this.commentField.getInput()) {
            this.session.comment = this.commentField.getInput()
        }

        this.sessionCache.setSession(this.session).then()
        this.onToggleEdit()
    }

    private handleSignIn(e: Event) {
        e.preventDefault()
        ROUTER.navigate(`/auth/sign_in?return_url=${encodeURIComponent(getPath())}`, {replace: true})
    }
}

export class SessionHeaderHandler {
    constructor() {
        ROUTER.route('session/:uuid/header', [this.handleSessionHeader])
    }

    handleSessionHeader = (uuid: string) => {
        SCREEN.setView(new SessionHeaderView(uuid))
    }
}
