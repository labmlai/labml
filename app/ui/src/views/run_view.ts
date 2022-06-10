import {Run} from '../models/run'
import {Status} from "../models/status"
import {User} from '../models/user'
import {ROUTER, SCREEN} from '../app'
import {Weya as $, WeyaElement} from '../../../lib/weya/weya'
import {DataLoader} from "../components/loader"
import {BackButton, CustomButton, ShareButton} from "../components/buttons"
import {UserMessages} from "../components/user_messages"
import {RunHeaderCard} from "../analyses/experiments/run_header/card"
import {experimentAnalyses} from "../analyses/analyses"
import {Card} from "../analyses/types"
import CACHE, {RunCache, RunsListCache, RunStatusCache, UserCache} from "../cache/cache"
import mix_panel from "../mix_panel"
import {handleNetworkErrorInplace} from '../utils/redirect'
import {AwesomeRefreshButton} from '../components/refresh_button'
import {setTitle} from '../utils/document'
import {ScreenView} from '../screen_view'

class RunView extends ScreenView {
    uuid: string
    run: Run
    runCache: RunCache
    status: Status
    statusCache: RunStatusCache
    runListCache: RunsListCache
    user: User
    userCache: UserCache
    actualWidth: number
    elem: HTMLDivElement
    runHeaderCard: RunHeaderCard
    cards: Card[] = []
    lastUpdated: number
    buttonsContainer: HTMLSpanElement
    private cardContainer: HTMLDivElement
    private loader: DataLoader
    private refresh: AwesomeRefreshButton
    private userMessages: UserMessages
    private share: ShareButton

    constructor(uuid: string) {
        super()
        this.uuid = uuid
        this.runCache = CACHE.getRun(this.uuid)
        this.statusCache = CACHE.getRunStatus(this.uuid)
        this.userCache = CACHE.getUser()
        this.runListCache = CACHE.getRunsList()

        this.userMessages = new UserMessages()

        this.loader = new DataLoader(async (force) => {
            this.status = await this.statusCache.get(force)
            this.run = await this.runCache.get(force)
            this.user = await this.userCache.get(force)
        })
        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))
        this.share = new ShareButton({
            text: 'run',
            parent: this.constructor.name
        })

        mix_panel.track('Run View', {uuid: this.uuid})
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
        setTitle({section: 'Run'})
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', '.run.page',
                {style: {width: `${this.actualWidth}px`}}, $ => {
                    $('div', $ => {
                        this.userMessages.render($)
                        $('div.nav-container', $ => {
                            new BackButton({text: 'Runs', parent: this.constructor.name}).render($)
                            this.refresh.render($)
                            this.buttonsContainer = $('span', '.float-right')
                            this.share.render($)
                        })
                        this.runHeaderCard = new RunHeaderCard({
                            uuid: this.uuid,
                            width: this.actualWidth,
                            lastUpdated: this.lastUpdated,
                            clickable: true
                        })
                        this.loader.render($)
                        this.runHeaderCard.render($)
                        this.cardContainer = $('div')
                    })
                })
        })

        try {
            await this.loader.load()

            setTitle({section: 'Run', item: this.run.name})
            this.renderButtons()
            this.renderClaimMessage()
            this.share.text = `${this.run.name} run`
            this.renderCards()
        } catch (e) {
            handleNetworkErrorInplace(e)
        } finally {
            if (this.status && this.status.isRunning) {
                this.refresh.attachHandler(this.runHeaderCard.renderLastRecorded.bind(this.runHeaderCard))
                this.refresh.start()
            }
        }
    }

    renderButtons() {
        this.buttonsContainer.innerHTML = ''
        $(this.buttonsContainer, $ => {
            if (!this.run.is_claimed) {
                new CustomButton({
                    onButtonClick: this.onRunAction.bind(this, true),
                    text: 'Claim',
                    title: 'own this run',
                    parent: this.constructor.name
                }).render($)
            } else if (!this.run.is_project_run || !this.user.is_complete) {
                new CustomButton({
                    onButtonClick: this.onRunAction.bind(this, false),
                    text: 'Add',
                    title: 'add to runs list',
                    parent: this.constructor.name
                }).render($)
            }
        })
    }

    renderClaimMessage() {
        if (!this.run.is_claimed) {
            this.userMessages.warning('This run will be deleted in 12 hours. Click Claim button to add it to your runs.')
        }
    }

    async onRunAction(isRunClaim: boolean) {
        if (!this.user.is_complete) {
            ROUTER.navigate(`/login?return_url=${window.location.pathname}`)
        } else {
            try {
                if (isRunClaim) {
                    await this.runListCache.claimRun(this.run)
                    this.userMessages.success('Successfully claimed and added to your runs list')
                    this.run.is_claimed = true
                } else {
                    await this.runListCache.addRun(this.run)
                    this.userMessages.success('Successfully added to your runs list')
                }
                this.run.is_project_run = true
                this.renderButtons()
            } catch (e) {
                this.userMessages.networkError()
                return
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
        let oldest = (new Date()).getTime()
        try {
            await this.loader.load(true)
        } catch (e) {

        } finally {
            if (this.status && !this.status.isRunning) {
                this.refresh.stop()
            }
            for (let card of this.cards) {
                await card.refresh()

                let lastUpdated = card.getLastUpdated()
                if (lastUpdated < oldest) {
                    oldest = lastUpdated
                }
            }

            this.lastUpdated = oldest
            await this.runHeaderCard.refresh(this.lastUpdated).then()
        }
    }

    onVisibilityChange() {
        this.refresh.changeVisibility(!document.hidden)
    }

    private renderCards() {
        $(this.cardContainer, $ => {
            experimentAnalyses.map((analysis, i) => {
                let card: Card = new analysis.card({uuid: this.uuid, width: this.actualWidth})
                this.cards.push(card)
                card.render($)
            })
        })
    }
}

export class RunHandler {
    constructor() {
        ROUTER.route('run/:uuid', [this.handleRun])
    }

    handleRun = (uuid: string) => {
        SCREEN.setView(new RunView(uuid))
    }
}
