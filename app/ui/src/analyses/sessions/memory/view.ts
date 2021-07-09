import {SeriesModel} from "../../../models/run"
import CACHE, {AnalysisDataCache, AnalysisPreferenceCache, SessionCache, SessionStatusCache} from "../../../cache/cache"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {Status} from "../../../models/status"
import {DataLoader} from "../../../components/loader"
import {ROUTER, SCREEN} from "../../../app"
import {BackButton, SaveButton} from "../../../components/buttons"
import {AnalysisPreferenceModel} from "../../../models/preferences"
import memoryCache from "./cache"
import {toPointValues} from "../../../components/charts/utils"
import {SessionHeaderCard} from '../session_header/card'
import {TimeSeriesChart} from '../../../components/charts/timeseries/chart'
import {SparkTimeLines} from '../../../components/charts/spark_time_lines/chart'
import mix_panel from "../../../mix_panel"
import {ViewHandler} from "../../types"
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {Session} from '../../../models/session'
import {setTitle} from '../../../utils/document'
import {ScreenView} from '../../../screen_view'

class MemoryView extends ScreenView {
    elem: HTMLDivElement
    uuid: string
    status: Status
    plotIdx: number[] = []
    statusCache: SessionStatusCache
    series: SeriesModel[]
    preferenceData: AnalysisPreferenceModel
    analysisCache: AnalysisDataCache
    preferenceCache: AnalysisPreferenceCache
    sessionHeaderCard: SessionHeaderCard
    sparkTimeLines: SparkTimeLines
    lineChartContainer: HTMLDivElement
    sparkLinesContainer: HTMLDivElement
    saveButtonContainer: HTMLDivElement
    saveButton: SaveButton
    isUpdateDisable: boolean
    actualWidth: number
    private loader: DataLoader
    private refresh: AwesomeRefreshButton
    private sessionCache: SessionCache
    private session: Session

    constructor(uuid: string) {
        super()

        this.uuid = uuid
        this.sessionCache = CACHE.getSession(this.uuid)
        this.statusCache = CACHE.getSessionStatus(this.uuid)
        this.analysisCache = memoryCache.getAnalysis(this.uuid)
        this.preferenceCache = memoryCache.getPreferences(this.uuid)

        this.isUpdateDisable = true
        this.saveButton = new SaveButton({onButtonClick: this.updatePreferences, parent: this.constructor.name})

        this.loader = new DataLoader(async (force) => {
            this.status = await this.statusCache.get(force)
            this.session = await this.sessionCache.get()
            this.series = toPointValues((await this.analysisCache.get(force)).series)
            this.preferenceData = await this.preferenceCache.get(force)
        })
        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))

        mix_panel.track('Analysis View', {uuid: this.uuid, analysis: this.constructor.name})
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
        setTitle({section: 'Memory'})
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', '.page',
                {style: {width: `${this.actualWidth}px`}},
                $ => {
                    $('div', $ => {
                        $('div', '.nav-container', $ => {
                            new BackButton({text: 'Session', parent: this.constructor.name}).render($)
                            this.saveButtonContainer = $('div')
                            this.refresh.render($)
                        })
                        this.sessionHeaderCard = new SessionHeaderCard({
                            uuid: this.uuid,
                            width: this.actualWidth
                        })
                        this.sessionHeaderCard.render($).then()
                        $('h2', '.header.text-center', 'Memory')
                        this.loader.render($)
                        $('div', '.detail-card', $ => {
                            this.lineChartContainer = $('div', '.fixed-chart')
                            this.sparkLinesContainer = $('div')
                        })
                    })
                })
        })

        try {
            await this.loader.load()

            setTitle({section: 'Memory', item: this.session.name})
            this.calcPreferences()

            this.renderSparkLines()
            this.renderLineChart()
            this.renderSaveButton()

        } catch (e) {
            handleNetworkErrorInplace(e)
        } finally {
            if (this.status && this.status.isRunning) {
                this.refresh.attachHandler(this.sessionHeaderCard.renderLastRecorded.bind(this.sessionHeaderCard))
                this.refresh.start()
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
        try {
            await this.loader.load(true)

            this.calcPreferences()
            this.renderSparkLines()
            this.renderLineChart()
        } catch (e) {

        } finally {
            if (this.status && !this.status.isRunning) {
                this.refresh.stop()
            }

            await this.sessionHeaderCard.refresh().then()
        }
    }

    onVisibilityChange() {
        this.refresh.changeVisibility(!document.hidden)
    }

    renderSaveButton() {
        this.saveButton.disabled = this.isUpdateDisable
        this.saveButtonContainer.innerHTML = ''
        $(this.saveButtonContainer, $ => {
            this.saveButton.render($)
        })
    }

    renderLineChart() {
        this.lineChartContainer.innerHTML = ''
        $(this.lineChartContainer, $ => {
            new TimeSeriesChart({
                series: this.series,
                width: this.actualWidth,
                plotIdx: this.plotIdx,
                onCursorMove: [this.sparkTimeLines.changeCursorValues],
                isCursorMoveOpt: true,
                isDivergent: true
            }).render($)
        })
    }

    renderSparkLines() {
        this.sparkLinesContainer.innerHTML = ''
        $(this.sparkLinesContainer, $ => {
            this.sparkTimeLines = new SparkTimeLines({
                series: this.series,
                plotIdx: this.plotIdx,
                width: this.actualWidth,
                onSelect: this.toggleChart,
                isDivergent: true
            })
            this.sparkTimeLines.render($)
        })
    }

    toggleChart = (idx: number) => {
        this.isUpdateDisable = false

        if (this.plotIdx[idx] >= 0) {
            this.plotIdx[idx] = -1
        } else {
            this.plotIdx[idx] = Math.max(...this.plotIdx) + 1
        }

        if (this.plotIdx.length > 1) {
            this.plotIdx = new Array<number>(...this.plotIdx)
        }

        this.renderSparkLines()
        this.renderLineChart()
        this.renderSaveButton()
    }

    calcPreferences() {
        if (this.isUpdateDisable) {
            let analysisPreferences = this.preferenceData.series_preferences
            if (analysisPreferences && analysisPreferences.length > 0) {
                this.plotIdx = [...analysisPreferences]
            } else if (this.series) {
                let res: number[] = []
                for (let i = 0; i < this.series.length; i++) {
                    res.push(i)
                }
                this.plotIdx = res
            }
        }
    }

    updatePreferences = () => {
        this.preferenceData.series_preferences = this.plotIdx
        this.preferenceCache.setPreference(this.preferenceData).then()

        this.isUpdateDisable = true
        this.renderSaveButton()
    }
}

export class MemoryHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('session/:uuid/memory', [this.handleMemory])
    }

    handleMemory = (uuid: string) => {
        SCREEN.setView(new MemoryView(uuid))
    }
}
