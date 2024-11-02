import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {ROUTER, SCREEN} from "../../../app"
import {Run} from "../../../models/run"
import CACHE, {DataStoreCache, RunCache, RunStatusCache, UserCache} from "../../../cache/cache"
import {Status} from "../../../models/status"
import {BackButton, DeleteButton, SaveButton,} from "../../../components/buttons"
import EditableField from "../../../components/input/editable_field"
import {formatTime, getTimeDiff} from "../../../utils/time"
import {DataLoader} from "../../../components/loader"
import {StatusView} from "../../../components/status"
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {setTitle} from '../../../utils/document'
import {formatFixed} from "../../../utils/value"
import {ScreenView} from '../../../screen_view'
import {User} from '../../../models/user'
import {UserMessages} from "../../../components/user_messages"
import {DataStore} from "../../../models/data_store";

enum EditStatus {
    NOCHANGE,
    SAVING,
    CHANGE
}

class RunHeaderView extends ScreenView {
    elem: HTMLDivElement
    run: Run
    runCache: RunCache
    status: Status
    statusCache: RunStatusCache
    user: User
    userCache: UserCache
    dataStoreCache: DataStoreCache
    dataStore: DataStore
    uuid: string
    actualWidth: number
    isProjectRun: boolean = false
    fieldContainer: HTMLDivElement
    computerButtonsContainer: HTMLSpanElement
    nameField: EditableField
    commentField: EditableField
    noteField: EditableField
    sizeField: EditableField
    tagField: EditableField
    dataStoreField: EditableField
    sizeCheckPoints: EditableField
    sizeTensorBoard: EditableField
    private deleteButton: DeleteButton
    private saveButton: SaveButton
    private loader: DataLoader

    constructor(uuid: string) {
        super()
        this.uuid = uuid
        this.runCache = CACHE.getRun(this.uuid)
        this.statusCache = CACHE.getRunStatus(this.uuid)
        this.userCache = CACHE.getUser()
        this.dataStoreCache = CACHE.getDataStore(this.uuid)

        this.deleteButton = new DeleteButton({onButtonClick: this.onDelete.bind(this), parent: this.constructor.name})
        this.loader = new DataLoader(async (force) => {
            this.status = await this.statusCache.get(force)
            this.run = await this.runCache.get(force)
            this.user = await this.userCache.get(force)
            this.dataStore = await this.dataStoreCache.get(force)
        })

        this.editStatus = EditStatus.NOCHANGE
    }

    get requiresAuth(): boolean {
        return false
    }

    set editStatus(value: EditStatus) {
        if (this.saveButton) {
            if (value === EditStatus.CHANGE) {
                this.saveButton.disabled = false
                this.saveButton.loading = false
            } else if (value === EditStatus.SAVING) {
                this.saveButton.disabled = true
                this.saveButton.loading = true
            } else {
                this.saveButton.disabled = true
                this.saveButton.loading = false
            }
        }
    }

    onResize(width: number) {
        super.onResize(width)

        this.actualWidth = Math.min(800, width)

        if (this.elem) {
            this._render().then()
        }
    }

    async _render() {
        setTitle({section: 'Run Details'})
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', '.page',
                {style: {width: `${this.actualWidth}px`}},
                $ => {
                    $('div', $ => {
                        $('div', '.nav-container', $ => {
                            new BackButton({text: 'Run', parent: this.constructor.name}).render($)
                            this.saveButton = new SaveButton({onButtonClick: this.updateRun, parent: this.constructor.name, isDisabled: true})
                            this.saveButton.render($)
                            this.deleteButton.render($)
                            this.deleteButton.hide(true)
                            this.computerButtonsContainer = $('span')
                        })
                        $('h2', '.header.text-center', 'Run Details')
                        this.loader.render($)
                        this.fieldContainer = $('div', '.input-list-container')
                    })
                })
        })

        try {
            await this.loader.load()

            setTitle({section: 'Run Details', item: this.run.name})
            this.renderFields()
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
                    value: this.run.name,
                    isEditable: true,
                    onChange: this.onInputChange.bind(this)
                })
                this.nameField.render($)
                new EditableField({
                    name: 'Rank',
                    value: this.run.rank,
                }).render($)
                new EditableField({
                    name: 'World Size',
                    value: this.run.world_size,
                }).render($)
                this.commentField = new EditableField({
                    name: 'Comment',
                    value: this.run.comment,
                    isEditable: true,
                    onChange: this.onInputChange.bind(this)
                })
                this.commentField.render($)
                this.tagField = new EditableField({
                    name: 'Tags',
                    value: this.run.tags.join(', '),
                    placeholder: 'Add tags separated by comma',
                    isEditable: true,
                    onChange: this.onInputChange.bind(this)
                })
                this.tagField.render($)
                this.noteField = new EditableField({
                    name: 'Note',
                    value: this.run.note,
                    placeholder: 'write your note here',
                    numEditRows: 5,
                    isEditable: true,
                    onChange: this.onInputChange.bind(this)
                })
                this.noteField.render($)
                $(`li`, $ => {
                    $('span', '.item-key', 'Run Status')
                    $('span', '.item-value', $ => {
                        new StatusView({status: this.status.run_status}).render($)
                    })
                })
                new EditableField({
                    name: 'UUID',
                    value: this.run.run_uuid,
                }).render($)
                new EditableField({
                    name: 'Start Time',
                    value: formatTime(this.run.start_time),
                }).render($)
                new EditableField({
                    name: 'Last Recorded',
                    value: this.status.isRunning ? getTimeDiff(this.status.last_updated_time * 1000) :
                        formatTime(this.status.last_updated_time),
                }).render($)
                new EditableField({
                    name: 'Start Step',
                    value: this.run.start_step
                }).render($)
                this.sizeField = new EditableField({
                    name: 'Size',
                    value: this.run.size ? formatFixed(this.run.size, 1) : '0'
                })
                this.sizeField.render($)
                this.sizeCheckPoints = new EditableField({
                    name: 'Checkpoints Size',
                    value: this.run.size_checkpoints ? formatFixed(this.run.size_checkpoints, 1) : '0'
                })
                this.sizeCheckPoints.render($)
                this.sizeTensorBoard = new EditableField({
                    name: 'TensorBoard Size',
                    value: this.run.size_tensorboard ? formatFixed(this.run.size_tensorboard, 1) : '0'
                })
                this.sizeTensorBoard.render($)
                new EditableField({
                    name: 'Python File',
                    value: this.run.python_file
                })
                $(`li`, $ => {
                    $('span', '.item-key', 'Remote Repo')
                    $('span', '.item-value', $ => {
                        $('a', this.run.repo_remotes, {
                            href: this.run.repo_remotes,
                            target: "_blank",
                            rel: "noopener noreferrer"
                        })
                    })
                })
                $(`li`, $ => {
                    $('span', '.item-key', 'Commit')
                    $('span', '.item-value', $ => {
                        $('a', this.run.commit, {
                            href: this.run.commit,
                            target: "_blank",
                            rel: "noopener noreferrer"
                        })
                    })
                })
                new EditableField({
                    name: 'Commit Message',
                    value: this.run.commit_message
                }).render($)

                this.dataStoreField = new EditableField({
                    name: 'Data Store',
                    value: JSON.stringify(this.dataStore.yamlString),
                    isEditable: true,
                    onChange: this.onInputChange.bind(this),
                    numEditRows: 10
                })
                this.dataStoreField.render($)
            })
        })
        this.deleteButton.hide(!(this.user.is_complete && this.run.is_claimed))
    }

    onInputChange(_: string) {
        this.editStatus = EditStatus.CHANGE
    }

    onDelete = async () => {
        if (confirm("Are you sure?")) {
            try {
                await CACHE.getRunsList().deleteRuns([this.uuid])
                ROUTER.navigate('/runs')
            } catch (e) {
                UserMessages.shared.networkError(e, "Failed to delete run")
                return
            }
        }
    }

    updateRun = async () => {
        this.editStatus = EditStatus.SAVING
        if (this.nameField.getInput()) {
            this.run.name = this.nameField.getInput()
        }

        if (this.commentField.getInput()) {
            this.run.comment = this.commentField.getInput()
        }

        if (this.noteField.getInput()) {
            this.run.note = this.noteField.getInput()
        }

        this.run.tags = this.tagField.getInput().split(',').map(tag => tag.trim())
        this.run.tags = this.run.tags.filter(tag => tag.length > 0)

        let data = {
            'name': this.run.name,
            'comment': this.run.comment,
            'note': this.run.note,
            'tags': this.run.tags
        }

        try {
            await this.runCache.updateRunData(data)
            await CACHE.getRunsList().localUpdateRun(this.run)
        } catch (e) {
            this.editStatus = EditStatus.CHANGE
            UserMessages.shared.networkError(e, "Failed to save run")
            return
        }

        let dataStore: any
        try {
            dataStore = JSON.parse(this.dataStoreField.getInput())
        } catch (e) {
            this.editStatus = EditStatus.CHANGE
            UserMessages.shared.error("Data Store is not a valid JSON")
            return
        }

        try {
            this.dataStore = await this.dataStoreCache.update(dataStore)
        } catch (e) {
            this.editStatus = EditStatus.CHANGE
            UserMessages.shared.networkError(e, "Failed to save data store")
            return
        }

        await this._render()
        this.editStatus = EditStatus.NOCHANGE
    }
}

export class RunHeaderHandler {
    constructor() {
        ROUTER.route('run/:uuid/header', [this.handleRunHeader])
    }

    handleRunHeader = (uuid: string) => {
        SCREEN.setView(new RunHeaderView(uuid))
    }
}
