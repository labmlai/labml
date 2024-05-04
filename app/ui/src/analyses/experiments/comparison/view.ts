import {Indicator, Run} from "../../../models/run"
import CACHE, {
    AnalysisDataCache,
    ComparisonAnalysisPreferenceCache
} from "../../../cache/cache"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {Status} from "../../../models/status"
import {DataLoader, Loader} from "../../../components/loader"
import {ROUTER, SCREEN} from "../../../app"
import {BackButton, DeleteButton, EditButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import {ComparisonPreferenceModel} from "../../../models/preferences"
import {fillPlotPreferences} from "../../../components/charts/utils"
import {ViewHandler} from "../../types"
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {clearChildElements, setTitle} from '../../../utils/document'
import {ScreenView} from '../../../screen_view'
import metricsCache from "./cache"
import {MetricDataStore, ViewWrapper} from "../chart_wrapper/view"
import comparisonCache from "./cache"
import {NetworkError} from "../../../network"
import {RunsPickerView} from "../../../views/run_picker_view"

class ComparisonView extends ScreenView implements MetricDataStore {
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
    trimSmoothEnds: boolean

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
    private loaderContainer: HTMLDivElement

    private actualWidth: number
    private refresh: AwesomeRefreshButton

    private loader: DataLoader
    private content: ViewWrapper

    private preferenceCache: ComparisonAnalysisPreferenceCache
    private run: Run
    private baseRun: Run
    private baseAnalysisCache: AnalysisDataCache
    private missingBaseExperiment: boolean
    private deleteButton: DeleteButton
    private editButton: EditButton

    constructor(uuid: string) {
        super()

        this.uuid = uuid
        this.chartType = 0
        this.preferenceCache = <ComparisonAnalysisPreferenceCache>comparisonCache.getPreferences(this.uuid)
        this.isUnsaved = false

        this.loader = new DataLoader(async (force) => {
            if (this.isUnsaved) {
                return
            }

            this.status = await CACHE.getRunStatus(this.uuid).get(force)
            this.run = await CACHE.getRun(this.uuid).get(force)

            metricsCache.getAnalysis(this.uuid).setCurrentUUID(this.uuid)
            this.series = (await metricsCache.getAnalysis(this.uuid).get(force)).series

            this.preferenceData = <ComparisonPreferenceModel>await this.preferenceCache.get(force)
            this.baseUuid = this.preferenceData.base_experiment

            this.chartType = this.preferenceData.chart_type
            this.stepRange = [...this.preferenceData.step_range]
            this.focusSmoothed = this.preferenceData.focus_smoothed
            this.plotIdx = [...fillPlotPreferences(this.series, this.preferenceData.series_preferences)]
            if (this.baseSeries) {
                this.basePlotIdx = [...fillPlotPreferences(this.baseSeries, this.preferenceData.base_series_preferences)]
            } else {
                this.basePlotIdx = [...this.preferenceData.base_series_preferences]
            }
            this.smoothValue = this.preferenceData.smooth_value
            this.trimSmoothEnds = this.preferenceData.trim_smooth_ends

            if (!!this.baseUuid) {
                await this.updateBaseRun(force)
            } else {
                this.missingBaseExperiment = true
            }
        })

        this.runHeaderCard = new RunHeaderCard({
            uuid: this.uuid,
            width: this.actualWidth / 2,
            showRank: false
        })

        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))
        this.deleteButton = new DeleteButton({onButtonClick: this.onDelete, parent: this.constructor.name})
        this.editButton = new EditButton({
            onButtonClick: this.onEditClick,
            parent: this.constructor.name
        })
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
                                this.buttonContainer = $('div')
                                this.saveButtonContainer = $('div')
                            })
                            this.refresh.render($)
                        })
                        this.loader.render($)
                        this.headerContainer = $('div', '.compare-header')
                        this.loaderContainer = $('div')
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
        this.content.render(this.missingBaseExperiment)
        this.renderButtons()
    }

    private setBaseLoading(value: boolean) {
        this.loaderContainer.innerHTML = ''
        if (value) {
            $(this.loaderContainer, $ => {
                new Loader().render($)
            })
            this.editButton.disabled = true
            this.deleteButton.disabled = true
            this.content.clear()
        } else {
            this.editButton.disabled = false
            this.deleteButton.disabled = false
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

            this.content.render(this.missingBaseExperiment)
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
        let preferenceData: ComparisonPreferenceModel = {
            series_preferences: this.plotIdx,
            chart_type: this.chartType,
            step_range: this.stepRange,
            focus_smoothed: this.focusSmoothed,
            sub_series_preferences: undefined,
            base_experiment: this.baseUuid,
            base_series_preferences: this.basePlotIdx,
            is_base_distributed: this.baseRun.world_size != 0,
            series_names: this.series.map(s => s.name),
            base_series_names: this.baseSeries ? this.baseSeries.map(s => s.name) : [],
            smooth_value: this.smoothValue,
            trim_smooth_ends: this.trimSmoothEnds
        }

        await this.preferenceCache.setPreference(preferenceData)

        this.isUnsaved = false
        this.refresh.resume()
    }

    private renderHeaders() {
        clearChildElements(this.headerContainer)
        $(this.headerContainer, $ => {
            this.runHeaderCard.render($).then()
            $('span', '.compared-with', $ => {
                $('span', '.sub', 'Compared With ')
                if (this.baseRun == null) {
                    $('span', '.title', 'No run selected')
                } else {
                    $('a', '.title.clickable', `${this.baseRun.name} `, {
                        on: {
                            click: () => {
                                window.open(`/run/${this.baseRun.run_uuid}`, '_blank')
                            }
                        }
                    })
                }
            })
        })
    }

    private onDelete = async () => {
        this.deleteButton.disabled = true
        this.deleteButton.loading = true
        try {
            this.preferenceData = await this.preferenceCache.deleteBaseExperiment()

            this.baseSeries = undefined
            this.basePlotIdx = []
            this.baseUuid = ''
            this.baseRun = undefined

            this.missingBaseExperiment = true
            this.isUnsaved = false
            this.refresh.resume()
            this.renderHeaders()
            this.content.render(this.missingBaseExperiment)
            this.renderButtons()
        } catch (e) {
            this.content.renderError(e, "Failed to delete comparison")
            this.deleteButton.disabled = false
        } finally {
            this.deleteButton.loading = false
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
                        this.preferenceData = await this.preferenceCache.updateBaseExperiment(run)
                        this.baseUuid = run.run_uuid
                        this.basePlotIdx = []
                        this.plotIdx = []
                        this.stepRange = [-1, -1]
                        this.focusSmoothed = true
                        this.smoothValue = 50
                        this.chartType = 0

                        this.setBaseLoading(true)
                        this.updateBaseRun(true).then(async () => {
                            this.isUnsaved = false
                            this.refresh.resume()

                            await this.savePreferences()
                            this.plotIdx = fillPlotPreferences(this.series, [])
                            this.basePlotIdx = fillPlotPreferences(this.baseSeries, [])

                            this.renderHeaders()
                            this.content.render(this.missingBaseExperiment)
                            this.renderButtons()

                            this.setBaseLoading(false)
                        })
                    }
                } catch (e) {
                    this.content.renderError(e, "Failed to update comparison")
                    this.setBaseLoading(false)
                } finally {
                    this.runPickerElem.classList.remove("fullscreen-cover")
                    document.body.classList.remove("stop-scroll")
                    clearChildElements(this.runPickerElem)
                }

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
        this.deleteButton.disabled = !this.baseUuid
        $(this.buttonContainer, $ => {
            this.deleteButton.render($)
            this.editButton.render($)
        })
    }

    private async updateBaseRun(force: boolean) {
        this.baseAnalysisCache = comparisonCache.getAnalysis(this.baseUuid)
        this.baseAnalysisCache.setCurrentUUID(this.uuid)
        this.baseRun = await CACHE.getRun(this.baseUuid).get(force)
        try {
            this.baseSeries = (await this.baseAnalysisCache.get(force)).series
            this.preferenceData.base_series_preferences = [...fillPlotPreferences(this.baseSeries, this.preferenceData.base_series_preferences)]
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
        this.series = (await metricsCache.getAnalysis(this.uuid).getAllMetrics()).series

        if (this.baseUuid !== '') {
            this.baseSeries = (await metricsCache.getAnalysis(this.baseUuid).getAllMetrics()).series
        }
    }
}

export class ComparisonHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('run/:uuid/compare', [this.handleComparison])
    }

    handleComparison = (uuid: string) => {
        SCREEN.setView(new ComparisonView(uuid))
    }
}
