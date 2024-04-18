import {CustomMetric, Indicator, Run} from "../../../models/run"
import CACHE, {AnalysisPreferenceCache} from "../../../cache/cache"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {Status} from "../../../models/status"
import {DataLoader} from "../../../components/loader"
import {ROUTER, SCREEN} from "../../../app"
import {BackButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import {AnalysisPreferenceModel} from "../../../models/preferences"
import {fillPlotPreferences} from "../../../components/charts/utils"
import {ViewHandler} from "../../types"
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {setTitle} from '../../../utils/document'
import {ScreenView} from '../../../screen_view'
import metricsCache from "./cache"
import {MetricDataStore, ViewWrapper} from "../chart_wrapper/view"
import EditableField from "../../../components/input/editable_field"
import {formatTime} from "../../../utils/time"
import {NetworkError} from "../../../network";

class MetricsView extends ScreenView implements MetricDataStore {
    private readonly uuid: string
    private readonly metricUuid?: string

    private elem: HTMLDivElement
    private runHeaderCard: RunHeaderCard
    private lineChartContainer: HTMLDivElement
    private sparkLinesContainer: HTMLDivElement
    private saveButtonContainer: WeyaElement
    private optionRowContainer: WeyaElement
    private actualWidth: number
    private refresh: AwesomeRefreshButton

    private nameField: EditableField
    private descriptionField: EditableField
    private createdAtField: EditableField
    private detailsContainer: WeyaElement

    private loader: DataLoader
    private content: ViewWrapper

    private preferenceCache: AnalysisPreferenceCache
    private run: Run
    private preferenceData: AnalysisPreferenceModel
    private status: Status
    private customMetric?: CustomMetric

    series: Indicator[]
    baseSeries?: Indicator[]
    plotIdx: number[]
    basePlotIdx?: number[]
    chartType: number
    focusSmoothed: boolean
    stepRange: number[]
    smoothValue: number
    isUnsaved: boolean

    constructor(uuid: string, metricUuid?: string) {
        super()

        this.uuid = uuid
        this.metricUuid = metricUuid
        this.chartType = 0
        this.preferenceCache = <AnalysisPreferenceCache>metricsCache.getPreferences(this.uuid)

        this.isUnsaved = false
        this.loader = new DataLoader(async (force) => {
            if (this.isUnsaved) {
                return
            }
            this.status = await CACHE.getRunStatus(this.uuid).get(force)
            this.run = await CACHE.getRun(this.uuid).get(force)
            this.series = (await metricsCache.getAnalysis(this.uuid).get(force)).series

            if (this.metricUuid != null) {
                let customMetricList = await CACHE.getCustomMetrics(this.uuid).get(force)
                if (customMetricList == null || customMetricList.getMetric(this.metricUuid) == null) {
                    throw new NetworkError(404, "")
                }

                this.preferenceData = customMetricList.getMetric(this.metricUuid).preferences
                this.customMetric = customMetricList.getMetric(this.metricUuid)
            } else {
                this.preferenceData = await this.preferenceCache.get(force)
            }

            this.chartType = this.preferenceData.chart_type
            this.stepRange = [...this.preferenceData.step_range]
            this.focusSmoothed = this.preferenceData.focus_smoothed
            this.plotIdx = [...fillPlotPreferences(this.series, this.preferenceData.series_preferences)]
            this.smoothValue = this.preferenceData.smooth_value
        })

        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))

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
        setTitle({section: 'Metrics'})
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
                        this.detailsContainer = $('div', '.input-list-container', $ => {
                        })

                        this.optionRowContainer = $('div')
                        $('h2', '.header.text-center', 'Metrics')
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
                preferenceChange: this.onPreferenceChange,
                deleteChart: this.customMetric == null ? null : this.onDelete
            })

            this.content.render()
            this.renderDetails()
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

    private onDelete = () => {
        // get confirmation from an alert
        if (confirm('Are you sure you want to delete this chart?')) {
            CACHE.getCustomMetrics(this.uuid).deleteMetric(this.metricUuid).then(() => {
                ROUTER.navigate(`/run/${this.uuid}`)
            })
        }
    }

    private onDetailChange = (text: string) => {
        this.content.onNonChartChange()
    }

    private renderDetails() {
        if (this.customMetric == null) {
            return
        }

        this.nameField = new EditableField({
            name: 'Name',
            value: this.customMetric.name,
            isEditable: true,
            onChange: this.onDetailChange
        })
        this.descriptionField = new EditableField({
            name: 'Description',
            value: this.customMetric.description,
            isEditable: true,
            numEditRows: 3,
            onChange: this.onDetailChange
        })
        this.createdAtField = new EditableField({
            name: 'Created at',
            value: formatTime(this.customMetric.createdTime),
            isEditable: false,
            onChange: this.onDetailChange
        })

        this.detailsContainer.innerHTML =  ''
        $(this.detailsContainer, $ => {
            $('ul', $ => {
                this.nameField.render($)
                this.descriptionField.render($)
                this.createdAtField.render($)
            })
        })

    }

    private async requestMissingMetrics() {
        this.series = (await metricsCache.getAnalysis(this.uuid).getAllMetrics()).series
    }

    private onPreferenceChange = () => {
        this.refresh.pause()
    }

    private savePreferences = async () => {
        let preferenceData: AnalysisPreferenceModel = {
            series_preferences: this.plotIdx,
            chart_type: this.chartType,
            step_range: this.stepRange,
            focus_smoothed: this.focusSmoothed,
            sub_series_preferences: undefined,
            series_names: this.series.map(s => s.name),
            smooth_value: this.smoothValue
        }

        if (this.metricUuid == null) {
            await this.preferenceCache.setPreference(preferenceData)
        } else {
            this.customMetric.preferences = preferenceData

            this.customMetric.name = this.nameField.getInput()
            this.customMetric.description = this.descriptionField.getInput()

            await CACHE.getCustomMetrics(this.uuid).updateMetric(this.metricUuid, this.customMetric.toData())
        }

        this.isUnsaved = false
        this.refresh.resume()
    }
}

export class MetricsHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('run/:uuid/metrics', [this.handleMetrics])
        ROUTER.route('run/:uuid/metrics/:metricUuid', [this.handleCustomMetrics])
    }

    handleMetrics = (uuid: string) => {
        SCREEN.setView(new MetricsView(uuid))
    }

    handleCustomMetrics = (uuid: string, metricUuid: string) => {
        SCREEN.setView(new MetricsView(uuid, metricUuid))
    }
}
