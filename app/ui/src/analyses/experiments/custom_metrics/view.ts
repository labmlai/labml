import {CustomMetric, Indicator, Run} from "../../../models/run"
import CACHE, {AnalysisDataCache} from "../../../cache/cache"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {Status} from "../../../models/status"
import {DataLoader, Loader} from "../../../components/loader"
import {ROUTER, SCREEN} from "../../../app"
import {BackButton, DeleteButton, IconButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import {ComparisonPreferenceModel} from "../../../models/preferences"
import {fillPlotPreferences} from "../../../components/charts/utils"
import {ViewHandler} from "../../types"
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {clearChildElements, setTitle} from '../../../utils/document'
import {ScreenView} from '../../../screen_view'
import {MetricDataStore, ViewWrapper} from "../chart_wrapper/view"
import {NetworkError} from "../../../network"
import {RunsPickerView} from "../../../views/run_picker_view"
import {RunsListItemView} from "../../../components/runs_list_item"
import metricsCache, {MetricCache} from "./cache"
import {SmoothingType} from "../../../components/charts/smoothing/smoothing_base";
import EditableField from "../../../components/input/editable_field";
import {formatTime} from "../../../utils/time";

class CustomMetricView extends ScreenView implements MetricDataStore {
    private readonly uuid: string
    private baseUuid: string

    private preferenceData: ComparisonPreferenceModel
    private status: Status

    series: Indicator[]
    baseSeries?: Indicator[]
    plotIdx: number[]
    basePlotIdx?: number[]
    chartType: number
    focusSmoothed: boolean
    stepRange: number[]
    smoothValue: number
    smoothFunction: string

    isUnsaved: boolean

    private elem: HTMLDivElement
    private runHeaderCard: RunHeaderCard
    private lineChartContainer: HTMLDivElement
    private sparkLinesContainer: HTMLDivElement
    private buttonContainer: HTMLDivElement
    private optionRowContainer: HTMLDivElement
    private messageContainer: HTMLDivElement
    private runPickerElem: HTMLDivElement
    private headerContainer: HTMLDivElement
    private saveButtonContainer: HTMLDivElement
    private topButtonContainer: HTMLDivElement
    private loaderContainer: HTMLDivElement

    private actualWidth: number
    private refresh: AwesomeRefreshButton

    private loader: DataLoader
    private content: ViewWrapper

    private run: Run
    private baseRun: Run
    private baseAnalysisCache: MetricCache
    private missingBaseExperiment: boolean
    private deleteButton: DeleteButton
    private editButton: IconButton
    private removeComparisonButton: IconButton

    private readonly customMetricUUID: string
    private customMetric: CustomMetric

    private nameField: EditableField
    private descriptionField: EditableField
    private createdAtField: EditableField
    private detailsContainer: WeyaElement

    constructor(uuid: string, customMetricUUID: string) {
        super()

        this.customMetricUUID = customMetricUUID
        this.uuid = uuid
        this.chartType = 0
        this.isUnsaved = false

        this.loader = new DataLoader(async (force) => {
            if (this.isUnsaved) {
                return
            }

            this.status = await CACHE.getRunStatus(this.uuid).get(force)
            this.run = await CACHE.getRun(this.uuid).get(force)

            let customMetricList = await CACHE.getCustomMetrics(this.uuid).get(force)
            if (customMetricList == null || customMetricList.getMetric(this.customMetricUUID) == null) {
                throw new NetworkError(404, "", "Custom metric list is null")
            }

            this.preferenceData = customMetricList.getMetric(this.customMetricUUID).preferences
            this.customMetric = customMetricList.getMetric(this.customMetricUUID)

            this.series = (await metricsCache.getAnalysis(this.uuid).get(force, this.preferenceData.series_preferences)).series

            this.baseUuid = this.preferenceData.base_experiment

            this.chartType = this.preferenceData.chart_type
            this.stepRange = [...this.preferenceData.step_range]
            this.focusSmoothed = this.preferenceData.focus_smoothed

            this.smoothValue = this.preferenceData.smooth_value
            this.smoothFunction = this.preferenceData.smooth_function

            if (!!this.baseUuid) {
                await this.updateBaseRun(force)
            } else {
                this.missingBaseExperiment = true
            }

            this.updatePlotIdxFromSeries()
        })

        this.runHeaderCard = new RunHeaderCard({
            uuid: this.uuid,
            width: this.actualWidth / 2,
            showRank: false
        })

        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))
        this.deleteButton = new DeleteButton({onButtonClick: this.onDelete, parent: this.constructor.name})
        this.editButton = new IconButton({
            onButtonClick: this.onEditClick,
            parent: this.constructor.name,
        }, '.fa.fa-balance-scale')
        this.removeComparisonButton = new IconButton({
            onButtonClick: this.onComparisonRemove,
            parent: this.constructor.name,
        }, '.fa.fa-times')

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
        setTitle({section: 'Comparison'})
        clearChildElements(this.elem)
        $(this.elem, $ => {
            this.runPickerElem = $('div')
            $('div', '.page',
                {style: {width: `${this.actualWidth}px`}},
                $ => {
                    $('div', $ => {
                        this.messageContainer = $('div')
                        $('div', '.nav-container', $ => {
                            new BackButton({text: 'Run', parent: this.constructor.name}).render($)

                            $('div', $ => {
                                this.saveButtonContainer = $('div')
                                this.topButtonContainer = $('div')
                            })
                            this.refresh.render($)
                        })
                        this.loader.render($)
                        this.headerContainer = $('div', '.compare-header')
                        this.loaderContainer = $('div')
                        this.detailsContainer = $('div')
                        this.optionRowContainer = $('div')
                        if (this.baseRun != null) {
                            $('h2', '.header.text-center', 'Comparison')
                        }
                        $('div', '.detail-card', $ => {
                            this.lineChartContainer = $('div', '.fixed-chart')
                            this.sparkLinesContainer = $('div')
                        })
                    })
                })
        })

        try {
            await this.loader.load()

            setTitle({section: 'Comparison', item: this.run.name})

            this.renderContent()
            this.renderDetails()
        } catch (e) {
            if (e.statusCode == 404) {
                this.missingBaseExperiment = true
                this.loader.removeErrorMessage()
                this.renderContent()
                let message = 'It appears the base run is missing or deleted. Please select another run to continue.'
                let errorMessage = $('h6', '.error', message)
                this.headerContainer.appendChild(errorMessage)
            } else {
                handleNetworkErrorInplace(e)
            }
        } finally {
            if (this.status && this.status.isRunning) {
                this.refresh.attachHandler(this.runHeaderCard.renderLastRecorded.bind(this.runHeaderCard))
                this.refresh.start()
            }
        }
    }

    renderContent() {
        this.content = new ViewWrapper({
            dataStore: this,
            lineChartContainer: this.lineChartContainer,
            sparkLinesContainer: this.sparkLinesContainer,
            saveButtonContainer: this.saveButtonContainer,
            optionRowContainer: this.optionRowContainer,
            messageContainer: this.messageContainer,
            actualWidth: this.actualWidth,
            requestMissingMetrics: this.requestMissingMetrics.bind(this),
            savePreferences: this.savePreferences.bind(this),
            preferenceChange: this.onPreferenceChange
        })

        this.renderHeaders()
        this.content.render()
        this.renderButtons()
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

    private onDetailChange = () => {
        this.content.onNonChartChange()
    }

    private setBaseLoading(value: boolean) {
        this.loaderContainer.innerHTML = ''
        if (value) {
            $(this.loaderContainer, $ => {
                new Loader().render($)
            })
            this.editButton.disabled = true
            this.removeComparisonButton.disabled = true
            this.content.clear()
        } else {
            this.editButton.disabled = false
            this.removeComparisonButton.disabled = false
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
            this.renderButtons()
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

    private onPreferenceChange = () => {
        this.refresh.pause()
    }

    private savePreferences = async () => {
        this.updateSeriesPreferencesFromPlotIdx()

        this.customMetric.preferences = {
            series_preferences: this.preferenceData.series_preferences,
            chart_type: this.chartType,
            step_range: this.stepRange,
            focus_smoothed: this.focusSmoothed,
            base_experiment: this.baseUuid,
            base_series_preferences: this.preferenceData.base_series_preferences,
            is_base_distributed: this.baseRun?.world_size != 0,
            smooth_value: this.smoothValue,
            smooth_function: this.smoothFunction
        }

        this.customMetric.name = this.nameField.getInput()
        this.customMetric.description = this.descriptionField.getInput()

        await CACHE.getCustomMetrics(this.uuid).updateMetric(this.customMetric.toData())

        this.isUnsaved = false
        this.refresh.resume()
    }

    private renderHeaders() {
        clearChildElements(this.headerContainer)
        $(this.headerContainer, $ => {
            this.runHeaderCard.render($).then()
            $('span', '.compared-with', $ => {
                $('span', '.sub', 'Compared With ')
                this.buttonContainer = $('div')
            })

            if (this.baseRun == null) {
                $('span', '.title', 'No run selected')
            } else {
                $('div.list.runs-list.list-group', $ => {
                    new RunsListItemView({
                        item: {
                            run_uuid: this.baseRun.run_uuid,
                            computer_uuid: this.baseRun.computer_uuid,
                            run_status: null,
                            last_updated_time: null,
                            name: this.baseRun.name,
                            comment: this.baseRun.comment,
                            start_time: this.baseRun.start_time,
                            world_size: this.baseRun.world_size,
                            metric_values: [],
                            step: null,
                            tags: this.baseRun.tags
                        },
                        width: this.actualWidth
                    })
                        .render($)
                })
            }
        })
    }

    private onComparisonRemove = async () => {
        this.removeComparisonButton.disabled = true
        this.removeComparisonButton.loading = true

        try {
            this.preferenceData.base_experiment = ''
            this.preferenceData.base_series_preferences = []
            this.baseSeries = undefined
            this.basePlotIdx = []
            this.baseUuid = ''
            this.baseRun = undefined

            await this.savePreferences()
        } catch (e) {
            this.content.renderError(e, "Failed to delete comparison")
            this.removeComparisonButton.disabled = false
            return
        } finally {
            this.removeComparisonButton.loading = false
        }

        this.isUnsaved = false

        this.refresh.resume()
        this.renderHeaders()
        this.content.render()
        this.renderButtons()
    }

    private onDelete = async () => {
        if (confirm('Are you sure you want to delete this chart?')) {
            try {
                await CACHE.getCustomMetrics(this.uuid).deleteMetric(this.customMetricUUID)
                ROUTER.navigate(`/run/${this.uuid}`)
            } catch (e) {
                this.content.renderError(e, "Failed to delete chart")
            }
        }
    }

    private onEditClick = () => {
        clearChildElements(this.runPickerElem)
        this.runPickerElem.classList.add("fullscreen-cover")
        document.body.classList.add("stop-scroll")
        this.runPickerElem.append(new RunsPickerView({
            title: 'Select run for comparison',
            excludedRuns: new Set<string>([this.run.run_uuid]),
            onPicked: async run => {
                try {
                    if (this.preferenceData.base_experiment !== run.run_uuid) {
                        this.baseUuid = run.run_uuid
                        this.basePlotIdx = []
                        this.plotIdx = []
                        this.stepRange = [-1, -1]
                        this.focusSmoothed = true
                        this.smoothValue = 0.5
                        this.chartType = 0
                        this.smoothFunction = SmoothingType.LEFT_EXPONENTIAL

                        this.setBaseLoading(true)

                        await this.updateBaseRun(true)

                        await this.savePreferences()

                        this.updatePlotIdxFromSeries()

                        this.isUnsaved = false
                        this.refresh.resume()
                    }
                } catch (e) {
                    this.content.renderError(e, "Failed to update comparison")
                    this.setBaseLoading(false)
                    return
                } finally {
                    this.runPickerElem.classList.remove("fullscreen-cover")
                    document.body.classList.remove("stop-scroll")
                    clearChildElements(this.runPickerElem)
                }

                this.renderHeaders()
                this.content.render()
                this.renderButtons()

                this.setBaseLoading(false)
            }, onCancel: () => {
                this.runPickerElem.classList.remove("fullscreen-cover")
                document.body.classList.remove("stop-scroll")
                clearChildElements(this.runPickerElem)
            }
        })
            .render())
    }

    private renderButtons() {
        clearChildElements(this.buttonContainer)
        this.removeComparisonButton.disabled = !this.baseUuid
        $(this.buttonContainer, $ => {
            this.removeComparisonButton.render($)
            this.editButton.render($)
        })

        clearChildElements(this.topButtonContainer)
        $(this.topButtonContainer, $ => {
            this.deleteButton.render($)
        })
    }

    private async updateBaseRun(force: boolean) {
        this.baseAnalysisCache = metricsCache.getAnalysis(this.baseUuid)
        this.baseRun = await CACHE.getRun(this.baseUuid).get(force)
        try {
            this.baseSeries = (await this.baseAnalysisCache.get(force, this.preferenceData.base_series_preferences)).series
            this.missingBaseExperiment = false
        } catch (e) {
            if (e instanceof NetworkError && e.statusCode === 404) {
                this.baseAnalysisCache = undefined
                this.baseSeries = undefined
                this.missingBaseExperiment = true
            } else {
                throw e
            }
        }
    }

    private async requestMissingMetrics() {
        this.updateSeriesPreferencesFromPlotIdx()

        let res = metricsCache.getAnalysis(this.uuid).get(false, this.preferenceData.series_preferences)
        this.series = (await res).series

        if (this.baseUuid !== '') {
            res = metricsCache.getAnalysis(this.baseUuid).get(false, this.preferenceData.base_series_preferences)
            this.baseSeries = (await res).series
        }

        this.updatePlotIdxFromSeries()
    }

    private updatePlotIdxFromSeries() {
        this.plotIdx = []
        for (let i = 0; i < this.series.length; i++) {
            this.plotIdx.push(this.preferenceData.series_preferences.includes(this.series[i].name) ? 1 : -1)
        }

        if (this.baseSeries) {
            this.basePlotIdx = []
            for (let i = 0; i < this.baseSeries.length; i++) {
                this.basePlotIdx.push(this.preferenceData.base_series_preferences.includes(this.baseSeries[i].name) ? 1 : -1)
            }
        }
    }

    private updateSeriesPreferencesFromPlotIdx() {
        this.plotIdx = fillPlotPreferences(this.series, this.plotIdx)

        let series_preferences = []
        for (let i = 0; i < this.series.length; i++) {
            if (this.plotIdx[i] != -1) {
                series_preferences.push(this.series[i].name)
            }
        }

        let base_series_preferences = []
        if (this.baseSeries) {
            this.basePlotIdx = fillPlotPreferences(this.baseSeries, this.basePlotIdx)
            for (let i = 0; i < this.baseSeries.length; i++) {
                if (this.basePlotIdx[i] != -1) {
                    base_series_preferences.push(this.baseSeries[i].name)
                }
            }
        }

        this.preferenceData.series_preferences = series_preferences
        this.preferenceData.base_series_preferences = base_series_preferences
    }

}

export class MetricHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('run/:uuid/metrics/:metric_uuid', [this.handleMetrics])
    }

    handleMetrics = (uuid: string, metric_uuid: string) => {
        SCREEN.setView(new CustomMetricView(uuid, metric_uuid))
    }
}
