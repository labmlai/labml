import {SeriesModel} from "../../../models/run"
import CACHE, {AnalysisPreferenceCache} from "../../../cache/cache"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {Status} from "../../../models/status"
import {DataLoader} from "../../../components/loader"
import {ROUTER, SCREEN} from "../../../app"
import {BackButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import {AnalysisPreferenceModel} from "../../../models/preferences"
import {toPointValues} from "../../../components/charts/utils"
import mix_panel from "../../../mix_panel"
import {ViewHandler} from "../../types"
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {setTitle} from '../../../utils/document'
import {ScreenView} from '../../../screen_view'
import metricsCache from "./cache"
import {DistributedViewContent, ViewContentData} from "../distributed_metrics/view"

class DistributedMetricsView extends ScreenView {
    uuid: string

    series: SeriesModel[]
    preferenceData: AnalysisPreferenceModel
    status: Status
    private plotIdx: number[] = []
    private currentChart: number
    private focusSmoothed: boolean
    private stepRange: number[]

    private elem: HTMLDivElement
    private runHeaderCard: RunHeaderCard
    private lineChartContainer: HTMLDivElement
    private sparkLinesContainer: HTMLDivElement
    private saveButtonContainer: WeyaElement
    private toggleButtonContainer: WeyaElement
    private isUpdateDisable: boolean
    private actualWidth: number
    private refresh: AwesomeRefreshButton

    private loader: DataLoader
    private content: DistributedViewContent
    private preferenceCache: AnalysisPreferenceCache

    constructor(uuid: string) {
        super()

        this.uuid = uuid
        this.currentChart = 0
        this.preferenceCache = metricsCache.getPreferences(this.uuid)

        this.isUpdateDisable = true
        this.loader = new DataLoader(async (force) => {
            this.status = await CACHE.getRunStatus(this.uuid).get(force)
            this.series = toPointValues((await metricsCache.getAnalysis(this.uuid).get(force)).series)
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
        setTitle({section: 'Distributed Metrics'})
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', '.page',
                {style: {width: `${this.actualWidth}px`}},
                $ => {
                    $('div', $ => {
                        $('div', '.nav-container', $ => {
                            new BackButton({text: 'Run', parent: this.constructor.name}).render($)
                            this.saveButtonContainer = $('div')
                            this.refresh.render($)
                        })
                        this.runHeaderCard = new RunHeaderCard({
                            uuid: this.uuid,
                            width: this.actualWidth,
                            showRank: false,
                        })
                        this.runHeaderCard.render($).then()
                        this.toggleButtonContainer = $('div.button-row')
                        $('h2', '.header.text-center', 'Distributed Metrics')
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

            // setTitle({section: 'Metrics', item: this.run.name})

            this.content = new DistributedViewContent({
                updatePreferences: this.updatePreferences,
                lineChartContainer: this.lineChartContainer,
                sparkLinesContainer: this.sparkLinesContainer,
                saveButtonContainer: this.saveButtonContainer,
                toggleButtonContainer: this.toggleButtonContainer,
                actualWidth: this.actualWidth,
                isUpdateDisable: this.isUpdateDisable
            })

            this.calcPreferences()

            this.content.render()
        } catch (e) {
            handleNetworkErrorInplace(e)
        } finally {
            if (this.status && this.status.isRunning) {
                this.refresh.attachHandler(this.runHeaderCard.renderLastRecorded.bind(this.runHeaderCard))
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
            this.content.render()
        } catch (e) {

        } finally {
            if (this.status && !this.status.isRunning) {
                this.refresh.stop()
            }
            await this.runHeaderCard.refresh().then()
        }
    }

    onVisibilityChange() {
        this.refresh.changeVisibility(!document.hidden)
    }

    private calcPreferences() {
        if(this.isUpdateDisable) {
            this.currentChart = this.preferenceData.chart_type
            this.stepRange = this.preferenceData.step_range
            this.focusSmoothed = this.preferenceData.focus_smoothed
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

            this.content.updateData({
                series: this.series,
                plotIdx: this.plotIdx,
                currentChart: this.currentChart,
                focusSmoothed: this.focusSmoothed,
                stepRange: this.stepRange
            })
        }
    }

    updatePreferences = (data: ViewContentData) => {
        this.plotIdx = data.plotIdx
        this.currentChart = data.currentChart
        this.focusSmoothed = data.focusSmoothed
        this.stepRange = data.stepRange

        this.preferenceData.series_preferences = this.plotIdx
        this.preferenceData.chart_type = this.currentChart
        this.preferenceData.step_range = this.stepRange
        this.preferenceData.focus_smoothed = this.focusSmoothed
        this.preferenceCache.setPreference(this.preferenceData).then()

        this.isUpdateDisable = true
        this.content.renderSaveButton()
    }
}

export class MergedDistributedMetricsHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('run/:uuid/merged_distributed', [this.handleMetrics])
    }

    handleMetrics = (uuid: string) => {
        SCREEN.setView(new DistributedMetricsView(uuid))
    }
}
