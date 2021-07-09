import {Status} from "../models/status"
import {IsUserLogged} from '../models/user'
import {ROUTER, SCREEN} from '../app'
import {Weya as $, WeyaElement} from '../../../lib/weya/weya'
import {DataLoader} from "../components/loader"
import {BackButton, CustomButton, ShareButton} from "../components/buttons"
import {Card} from "../analyses/types"
import CACHE, {IsUserLoggedCache, SessionCache, SessionsListCache, SessionStatusCache} from "../cache/cache"
import {Session} from '../models/session'
import {SessionHeaderCard} from '../analyses/sessions/session_header/card'
import {sessionAnalyses} from '../analyses/analyses'
import {UserMessages} from "../components/user_messages"
import mix_panel from "../mix_panel"
import {handleNetworkErrorInplace} from '../utils/redirect'
import {AwesomeRefreshButton} from '../components/refresh_button'
import {setTitle} from '../utils/document'
import {ScreenView} from '../screen_view'

class SessionView extends ScreenView {
    uuid: string
    session: Session
    sessionCache: SessionCache
    status: Status
    statusCache: SessionStatusCache
    sessionsListCache: SessionsListCache
    isUserLogged: IsUserLogged
    isUserLoggedCache: IsUserLoggedCache
    actualWidth: number
    elem: HTMLDivElement
    sessionHeaderCard: SessionHeaderCard
    cards: Card[] = []
    lastUpdated: number
    ButtonsContainer: HTMLSpanElement
    private cardContainer: HTMLDivElement
    private loader: DataLoader
    private refresh: AwesomeRefreshButton
    private userMessages: UserMessages
    private share: ShareButton

    constructor(uuid: string) {
        super()
        this.uuid = uuid
        this.sessionCache = CACHE.getSession(this.uuid)
        this.statusCache = CACHE.getSessionStatus(this.uuid)
        this.isUserLoggedCache = CACHE.getIsUserLogged()
        this.sessionsListCache = CACHE.getSessionsList()

        this.userMessages = new UserMessages()

        this.loader = new DataLoader(async (force) => {
            this.session = await this.sessionCache.get(force)
            this.status = await this.statusCache.get(force)
            this.isUserLogged = await this.isUserLoggedCache.get(force)
        })
        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))
        this.share = new ShareButton({
            text: 'computer',
            parent: this.constructor.name
        })

        mix_panel.track('Computer View', {uuid: this.uuid})
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
        setTitle({section: 'Computer'})
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', '.run.page',
                {style: {width: `${this.actualWidth}px`}}, $ => {
                    this.userMessages.render($)
                    this.ButtonsContainer = $('span', '.float-right')
                    $('div', '.nav-container', $ => {
                        new BackButton({text: 'Computers', parent: this.constructor.name}).render($)
                        this.refresh.render($)
                        this.share.render($)
                    })
                    this.sessionHeaderCard = new SessionHeaderCard({
                        uuid: this.uuid,
                        width: this.actualWidth,
                        lastUpdated: this.lastUpdated,
                        clickable: true
                    })
                    this.loader.render($)
                    this.sessionHeaderCard.render($)
                    this.cardContainer = $('div')
                })
        })

        try {
            await this.loader.load()

            setTitle({section: 'Computer', item: this.session.name})
            this.renderButtons()
            this.share.text = `${this.session.name} computer`
            this.renderCards()
        } catch (e) {
            handleNetworkErrorInplace(e)
        } finally {
            if (this.status && this.status.isRunning) {
                this.refresh.attachHandler(this.sessionHeaderCard.renderLastRecorded.bind(this.sessionHeaderCard))
                this.refresh.start()
            }
        }
    }

    renderButtons() {
        this.ButtonsContainer.innerHTML = ''
        $(this.ButtonsContainer, $ => {
            if (!this.session.is_claimed) {
                new CustomButton({
                    onButtonClick: this.onSessionAction.bind(this, true),
                    text: 'Claim',
                    parent: this.constructor.name
                }).render($)
            } else if (!this.session.is_project_session || !this.isUserLogged.is_user_logged) {
                new CustomButton({
                    onButtonClick: this.onSessionAction.bind(this, false),
                    text: 'Add',
                    parent: this.constructor.name
                }).render($)
            }
        })
    }

    async onSessionAction(isSessionClaim: boolean) {
        if (!this.isUserLogged.is_user_logged) {
            mix_panel.track('Claim Button Click', {uuid: this.uuid, analysis: this.constructor.name})
            ROUTER.navigate(`/login#return_url=${window.location.pathname}`)
        } else {
            try {
                if (isSessionClaim) {
                    await this.sessionsListCache.claimSession(this.session)
                    this.userMessages.success('Successfully claimed and added to your computers list')
                    this.session.is_claimed = true
                } else {
                    await this.sessionsListCache.addSession(this.session)
                    this.userMessages.success('Successfully added to your computers list')
                }

                this.session.is_project_session = true
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
            await this.sessionHeaderCard.refresh(this.lastUpdated).then()
        }
    }

    onVisibilityChange() {
        this.refresh.changeVisibility(!document.hidden)
    }

    private renderCards() {
        $(this.cardContainer, $ => {
            sessionAnalyses.map((analysis, i) => {
                let card: Card = new analysis.card({uuid: this.uuid, width: this.actualWidth})
                this.cards.push(card)
                card.render($)
            })
        })
    }
}

export class SessionHandler {
    constructor() {
        ROUTER.route('session/:uuid', [this.handleSession])
    }

    handleSession = (uuid: string) => {
        SCREEN.setView(new SessionView(uuid))
    }
}
