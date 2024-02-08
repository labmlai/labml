import {Run, SeriesModel} from "../../../models/run"
import CACHE, {AnalysisPreferenceCache} from "../../../cache/cache"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {Status} from "../../../models/status"
import {DataLoader} from "../../../components/loader"
import {ROUTER, SCREEN} from "../../../app"
import {BackButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import {AnalysisPreferenceModel} from "../../../models/preferences"
import {fillPlotPreferences, toPointValues} from "../../../components/charts/utils"
import mix_panel from "../../../mix_panel"
import {ViewHandler} from "../../types"
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {setTitle} from '../../../utils/document'
import {ScreenView} from '../../../screen_view'
import metricsCache from "./cache"
import {MetricDataStore, ViewWrapper} from "../chart_wrapper/view"

class DistributedMetricsView extends ScreenView implements MetricDataStore {
    private readonly uuid: string

    private elem: HTMLDivElement
    private runHeaderCard: RunHeaderCard
    private lineChartContainer: HTMLDivElement
    private sparkLinesContainer: HTMLDivElement
    private saveButtonContainer: WeyaElement
    private optionRowContainer: WeyaElement
    private actualWidth: number
    private refresh: AwesomeRefreshButton

    private loader: DataLoader
    private content: ViewWrapper

    private preferenceCache: AnalysisPreferenceCache
    private run: Run
    private preferenceData: AnalysisPreferenceModel
    private status: Status

    series: SeriesModel[]
    baseSeries?: SeriesModel[]
    plotIdx: number[]
    basePlotIdx?: number[]
    chartType: number
    focusSmoothed: boolean
    stepRange: number[]
    isUnsaved: boolean

    constructor(uuid: string) {
        super()

        this.uuid = uuid
        this.chartType = 0
        this.preferenceCache = <AnalysisPreferenceCache>metricsCache.getPreferences(this.uuid)

        this.isUnsaved = false
        this.loader = new DataLoader(async (force) => {
            if (this.isUnsaved) {
                return
            }
            this.run = await CACHE.getRun(this.uuid).get(force)
            this.status = await CACHE.getRunStatus(this.uuid).get(force)
            this.series = toPointValues((await metricsCache.getAnalysis(this.uuid).get(force)).series)
            this.preferenceData = await this.preferenceCache.get(force)

            this.chartType = this.preferenceData.chart_type
            this.stepRange = [...this.preferenceData.step_range]
            this.focusSmoothed = this.preferenceData.focus_smoothed
            this.plotIdx = [...fillPlotPreferences(this.series, this.preferenceData.series_preferences)]
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
                        this.optionRowContainer = $('div.button-row')
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

            setTitle({section: 'Metrics', item: this.run.name})

            this.content = new ViewWrapper({
                dataStore: this,
                lineChartContainer: this.lineChartContainer,
                sparkLinesContainer: this.sparkLinesContainer,
                saveButtonContainer: this.saveButtonContainer,
                optionRowContainer: this.optionRowContainer,
                actualWidth: this.actualWidth,
                requestMissingMetrics: this.requestMissingMetrics.bind(this),
                savePreferences: this.savePreferences.bind(this),
                preferenceChange: this.onPreferenceChange
            })

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
        if (this.isUnsaved) {
            this.refresh.pause()
            return
        } else {
            this.refresh.resume()
        }

        try {
            await this.loader.load(true)

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

    private async requestMissingMetrics() {
        metricsCache.getAnalysis(this.uuid).setMetricData(this.plotIdx)
        this.series = toPointValues((await metricsCache.getAnalysis(this.uuid).get(true)).series)
    }

    private onPreferenceChange = () => {
        this.refresh.pause()
    }

    private savePreferences = () => {
        let preferenceData: AnalysisPreferenceModel = {
            series_preferences: this.plotIdx,
            chart_type: this.chartType,
            step_range: this.stepRange,
            focus_smoothed: this.focusSmoothed,
            sub_series_preferences: undefined
        }

        this.preferenceCache.setPreference(preferenceData).then()
        this.isUnsaved = false
        this.refresh.resume()
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
