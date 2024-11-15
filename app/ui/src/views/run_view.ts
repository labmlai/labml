import {CustomMetricList, Run} from '../models/run'
import {Status} from "../models/status"
import {User} from '../models/user'
import {ROUTER, SCREEN} from '../app'
import {Weya as $, WeyaElement} from '../../../lib/weya/weya'
import {DataLoader} from "../components/loader"
import {AddButton, BackButton, CustomButton, ExpandButton, IconButton, ShareButton} from "../components/buttons"
import {UserMessages} from "../components/user_messages"
import {RunHeaderCard} from "../analyses/experiments/run_header/card"
import {experimentAnalyses} from "../analyses/analyses"
import {Card} from "../analyses/types"
import CACHE, {RunCache, RunStatusCache, UserCache} from "../cache/cache"
import {handleNetworkErrorInplace} from '../utils/redirect'
import {AwesomeRefreshButton} from '../components/refresh_button'
import {setTitle} from '../utils/document'
import {ScreenView} from '../screen_view'
import metricAnalysis from "../analyses/experiments/custom_metrics";
import NETWORK from "../network";

class RunView extends ScreenView {
    uuid: string
    rank?: string
    run: Run
    runCache: RunCache
    status: Status
    statusCache: RunStatusCache
    user: User
    userCache: UserCache
    actualWidth: number
    elem: HTMLDivElement
    runHeaderCard: RunHeaderCard
    cards: Card[] = []
    lastUpdated: number
    buttonsContainer: HTMLSpanElement
    private cardContainer: HTMLDivElement
    private rankContainer: WeyaElement
    private loader: DataLoader
    private refresh: AwesomeRefreshButton
    private share: ShareButton
    private addCustomMetricButton: AddButton
    private magicMetricButton: IconButton
    private isRankExpanded: boolean
    private rankElems: WeyaElement
    private processContainer: WeyaElement
    private customMetrics: CustomMetricList

    private progressText: HTMLElement

    constructor(uuid: string) {
        super()
        this.uuid = uuid
        this.rank = uuid.split('_')[1]
        this.runCache = CACHE.getRun(this.uuid)
        this.statusCache = CACHE.getRunStatus(this.uuid)
        this.userCache = CACHE.getUser()
        this.isRankExpanded = false

        this.loader = new DataLoader(async (force) => {
            this.reloadStatus = "Loading status"
            this.status = await this.statusCache.get(force)

            this.reloadStatus = "Loading status"
            this.run = await this.runCache.get(force)

            this.reloadStatus = "Loading user"
            this.user = await this.userCache.get(force)

            this.reloadStatus = "Loading charts"
            this.customMetrics = await CACHE.getCustomMetrics(this.uuid).get(force)

            this.reloadStatus = ""
        })
        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))
        this.share = new ShareButton({
            text: 'run',
            parent: this.constructor.name
        })
        this.addCustomMetricButton = new AddButton({
            onButtonClick: () => {
                this.createCustomMetric().then()
            },
            title: 'Add custom metric',
            parent: this.constructor.name
        })
        this.magicMetricButton = new IconButton({
            onButtonClick: () => {
                this.magicMetricButton.loading = true
                this.creatMagicMetric().then(() => {
                    this.magicMetricButton.loading = false
                })
            },
            title: 'Add magic metric',
            parent: this.constructor.name,
        }, '.fas.fa-magic')
    }

    private get isRank(): boolean {
        return this.run?.isRank || false
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
                        $('div.nav-container', $ => {
                            new BackButton({text: 'Runs', parent: this.constructor.name}).render($)


                            this.refresh.render($)
                            this.buttonsContainer = $('span', '.float-right')
                            this.share.render($)

                            this.progressText = $('span', '.progress-text.float-right')
                        })
                        this.runHeaderCard = new RunHeaderCard({
                            uuid: this.uuid,
                            width: this.actualWidth,
                            lastUpdated: this.lastUpdated,
                            clickable: !this.isRank,
                            showRank: !!this.isRank
                        })
                        this.loader.render($)
                        this.runHeaderCard.render($)
                        this.rankContainer = $('div.list.runs-list.list-group')
                        this.processContainer = $('div.fit-content.button-row.process-row')
                        this.cardContainer = $('div')
                    })
                })
        })

        try {
            await this.loader.load()

            setTitle({section: 'Run', item: this.run.name})
            this.renderButtons()
            this.renderClaimMessage()
            this.renderRanks()
            this.share.text = `${this.run.name} run`
            this.renderCards()
            this.renderProcess()
        } catch (e) {
            handleNetworkErrorInplace(e)
        } finally {
            if (this.status && this.status.isRunning) {
                this.refresh.attachHandler(this.runHeaderCard.renderLastRecorded.bind(this.runHeaderCard))
                this.refresh.start()
            }
        }
    }

    set reloadStatus(text: string) {
        if (this.progressText) {
            this.progressText.innerText = text
        }
    }

    renderProcess() {
        if (this.run == null) {
            return
        }
        this.processContainer.innerHTML = ''

        $(this.processContainer, $ => {
            new CustomButton({
                onButtonClick: () => {
                    ROUTER.navigate(`session/${this.run.session_id}/process/${this.run.process_id}`)
                },
                text: 'Process',
                title: 'Running process for the experiment',
                parent: this.constructor.name,
                isDisabled: this.run.process_id == ''
            }).render($)

            new CustomButton({
                onButtonClick: () => {
                    ROUTER.navigate(`session/${this.run.session_id}?from=run`)
                },
                text: 'Computer',
                title: 'Running computer for the experiment',
                parent: this.constructor.name,
                isDisabled: this.run.session_id == ''
            }).render($)

            new CustomButton({
                onButtonClick: () => {
                    window.open(this.run.commit, "_blank")
                },
                text: 'Source',
                title: 'Source repository for the experiment',
                parent: this.constructor.name,
                isDisabled: this.run.repo_remotes == ''
            }).render($)
        })
    }

    renderButtons() {
        this.buttonsContainer.innerHTML = ''
        $(this.buttonsContainer, $ => {
            if (!this.run.is_claimed && !this.isRank) {
                new CustomButton({
                    onButtonClick: this.onRunAction.bind(this, true),
                    text: 'Claim',
                    title: 'own this run',
                    parent: this.constructor.name
                }).render($)
            } else if ((!this.run.is_project_run || !this.user.is_complete) && !this.isRank) {
                new CustomButton({
                    onButtonClick: this.onRunAction.bind(this, false),
                    text: 'Add',
                    title: 'add to runs list',
                    parent: this.constructor.name
                }).render($)
            }
            this.addCustomMetricButton.render($)
            this.magicMetricButton.render($)
        })
    }

    renderClaimMessage() {
        if (!this.run.is_claimed && !this.isRank) {
            UserMessages.shared.warning('This run will be deleted in 12 hours. Click Claim button to add it to your runs.')
        }
    }

    async creatMagicMetric() {
        let res = null
        try {
            res = await NETWORK.createMagicMetric(this.uuid)
        } catch (e) {
            UserMessages.shared.networkError(e, 'Failed to create magic metric')
            return
        }


        if (res['is_success']) {
            try {
                this.customMetrics = await CACHE.getCustomMetrics(this.uuid).get(true)
            } catch (e) {
                UserMessages.shared.networkError(e, 'Failed to load custom metrics')
                return
            }

            this.renderCards()
            UserMessages.shared.success('Chart created')
        } else {
            UserMessages.shared.warning(res['message'])
        }
    }

    async createCustomMetric() {
        this.addCustomMetricButton.loading = true
        try {
            let customMetric = await CACHE.getCustomMetrics(this.uuid).createMetric({
                name: 'New Chart',
                description: ''
            })
            ROUTER.navigate(`/run/${this.uuid}/metrics/${customMetric.id}`)
        } catch (e) {
            UserMessages.shared.networkError(e, 'Failed to create custom metric')
        } finally {
            this.addCustomMetricButton.loading = false
        }
    }

    async onRunAction(isRunClaim: boolean) {
        if (!this.user.is_complete) {
            ROUTER.navigate(`/login?return_url=${window.location.pathname}`)
        } else {
            try {
                if (isRunClaim) {
                    await CACHE.getRunsList().claimRun(this.run)
                    UserMessages.shared.success('Successfully claimed and added to your runs list')
                    this.run.is_claimed = true
                } else {
                    await CACHE.getRunsList().addRun(this.run)
                    UserMessages.shared.success('Successfully added to your runs list')
                }
                this.run.is_project_run = true
            } catch (e) {
                UserMessages.shared.networkError(e, "Failed to claim the run")
                return
            }
            this.renderButtons()
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
                this.reloadStatus = `Refreshing ${card.cardName()}`
                await card.refresh()

                let lastUpdated = card.getLastUpdated()
                if (lastUpdated < oldest) {
                    oldest = lastUpdated
                }
            }

            this.lastUpdated = oldest
            this.reloadStatus = 'Updating Run header'
            await this.runHeaderCard.refresh(this.lastUpdated)
            this.reloadStatus = ''
        }
    }

    onVisibilityChange() {
        this.refresh.changeVisibility(!document.hidden)
    }

    private renderCards() {
        this.cardContainer.innerHTML = ''
        this.cards = []
        $(this.cardContainer, $ => {
            if (this.customMetrics != null && this.run != null) {
                let metricList = this.customMetrics.getMetrics()
                metricList.sort((a, b) => {
                    if (a.position == b.position) {
                        return b.createdTime - a.createdTime
                    }
                    return a.position - b.position
                })
                metricList.map((metric, i) => {
                    let card = new metricAnalysis.card({uuid: this.uuid, width: this.actualWidth, params: {
                            custom_metric: metric.id
                        }})
                    this.cards.push(card)
                    card.render($)
                })
            }

            experimentAnalyses.map((analysis, i) => {
                let card: Card = new analysis.card({uuid: this.uuid, width: this.actualWidth})
                this.cards.push(card)
                card.render($)
            })
        })
    }

    private renderRanks() {
        this.rankContainer.innerHTML = ''
        if (this.isRank || this.run.world_size == 0) { // no ranks or not the master
            return
        }
        $(this.rankContainer, $ => {
            $('div', '.toggle-list-title', $ => {
                $('h3.header', `Ranks`)
                $('hr')
                new ExpandButton({
                    onButtonClick: () => {
                        this.isRankExpanded = !this.isRankExpanded
                        this.rankElems.classList.toggle('hidden')
                    }, parent: this.constructor.name
                })
                    .render($)
            })
            this.rankElems = $('div', '.hidden.list.runs-list.list-group', $ => {
                for (const [rank, run_uuid] of Object.entries(this.run.other_rank_run_uuids)) {
                    $('a', '.list-item.list-group-item.list-group-item-action',
                        {href: `/run/${run_uuid}`, target: "_blank"},
                        $ => {
                            $('div', $ => {
                                $('h6', `Rank ${+rank + 1}`)
                            })
                    })
                }
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
