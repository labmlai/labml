import {Weya, WeyaElementFunction} from '../../../../../lib/weya/weya'
import {ROUTER} from '../../../app'
import CACHE, {RunCache, RunStatusCache} from "../../../cache/cache"
import {CardOptions} from "../../types"
import {Run} from "../../../models/run"
import {Status} from "../../../models/status"
import {StatusView} from "../../../components/status"
import {formatTime, getTimeDiff} from "../../../utils/time"
import {DataLoader} from "../../../components/loader"

interface RunHeaderOptions extends CardOptions {
    lastUpdated?: number
    clickable?: boolean
    showRank?: boolean
}

export class RunHeaderCard {
    run: Run
    uuid: string
    status: Status
    lastUpdated: number
    runCache: RunCache
    elem: HTMLDivElement
    lastRecordedContainer: HTMLDivElement
    statusViewContainer: HTMLDivElement
    statusCache: RunStatusCache
    private clickable: boolean
    private loader: DataLoader
    private showRank: boolean

    constructor(opt: RunHeaderOptions) {
        this.uuid = opt.uuid
        this.clickable = opt.clickable ?? false
        this.runCache = CACHE.getRun(this.uuid)
        this.statusCache = CACHE.getRunStatus(this.uuid)
        this.showRank = opt.showRank ?? true

        this.loader = new DataLoader(async (force) => {
            this.status = await this.statusCache.get(force)
            this.run = await this.runCache.get(force)

            this.lastUpdated = opt.lastUpdated ? opt.lastUpdated : this.statusCache.lastUpdated
        })
    }

    async render($: WeyaElementFunction) {
        this.elem = $('div', '.labml-card.disabled', $ => {
            this.loader.render($)
        })

        if (this.clickable) {
            this.elem.classList.remove('disabled')
            this.elem.classList.add('labml-card-action')
            this.elem.addEventListener('click', this.onClick)
        }

        try {
            await this.loader.load()

            Weya(this.elem, $ => {
                $('div', $ => {
                    $('div', $ => {
                        this.lastRecordedContainer = $('div', '.last-updated.mb-2')
                    })
                    $('div', '.run-info', $ => {
                        this.statusViewContainer = $('div')
                        $('h3', `${this.run.name}`)
                        $('h5', `${this.run.comment}`)
                    })

                    if (this.showRank && this.run.world_size > 0) {
                        $('div', '.rank.mt-2', $ => {
                            $('span', `Rank ${this.run.rank + 1} of ${this.run.world_size}`)
                        })
                    }
                })
            })

            this.renderStatusView()
            this.renderLastRecorded()
        } catch (e) {
            console.error(e)
        }
    }

    renderLastRecorded() {
        let lastRecorded = this.status.last_updated_time

        this.lastRecordedContainer.innerText = `Last Recorded ${this.status.isRunning ?
            getTimeDiff(lastRecorded * 1000) : 'on ' + formatTime(lastRecorded)}`
    }

    renderStatusView() {
        this.statusViewContainer.innerHTML = ''
        Weya(this.statusViewContainer, $ => {
            new StatusView({status: this.status.run_status}).render($)
        })
    }

    async refresh(lastUpdated?: number) {
        try {
            await this.loader.load(true)

            this.lastUpdated = lastUpdated ? lastUpdated : this.statusCache.lastUpdated

            this.renderStatusView()
            this.renderLastRecorded()

        } catch (e) {
        }
    }

    onClick = () => {
        ROUTER.navigate(`/run/${this.uuid}/header`)
    }
}
