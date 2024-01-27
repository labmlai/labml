import {Run, SeriesModel} from "../../../models/run"
import CACHE, {AnalysisDataCache, AnalysisPreferenceCache} from "../../../cache/cache"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {Status} from "../../../models/status"
import {DataLoader} from "../../../components/loader"
import {ROUTER, SCREEN} from "../../../app"
import {BackButton, DeleteButton, EditButton, SaveButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import {AnalysisPreferenceModel, ComparisonPreferenceModel} from "../../../models/preferences"
import {defaultSeriesToPlot, toPointValues} from "../../../components/charts/utils"
import mix_panel from "../../../mix_panel"
import {ViewHandler} from "../../types"
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {clearChildElements, setTitle} from '../../../utils/document'
import {ScreenView} from '../../../screen_view'
import metricsCache from "./cache"
import {ViewWrapper, ViewWrapperData} from "../chart_wrapper/view"
import comparisonCache from "./cache"
import {NetworkError} from "../../../network"
import {RunsPickerView} from "../../../views/run_picker_view";

// TODO missing base experiment
class ComparisonView extends ScreenView {
    private readonly uuid: string
    private baseUuid: string

    private currentSeries: SeriesModel[]
    private baseSeries: SeriesModel[]
    private preferenceData: ComparisonPreferenceModel
    private status: Status
    private currentPlotIdx: number[] = []
    private basePlotIdx: number[] = []
    private currentChart: number
    private focusSmoothed: boolean
    private stepRange: number[]

    private elem: HTMLDivElement
    private runHeaderCard: RunHeaderCard
    private lineChartContainer: HTMLDivElement
    private sparkLinesContainer: HTMLDivElement
    private buttonContainer: HTMLDivElement
    private toggleButtonContainer: HTMLDivElement
    private isUpdateDisable: boolean
    private actualWidth: number
    private refresh: AwesomeRefreshButton

    private loader: DataLoader
    private content: ViewWrapper
    private preferenceCache: AnalysisPreferenceCache
    private run: Run
    private runPickerElem: HTMLDivElement
    private headerContainer: HTMLDivElement
    private baseRun: Run
    private baseAnalysisCache: AnalysisDataCache
    private missingBaseExperiment: Boolean
    private deleteButton: DeleteButton
    private saveButtonContainer: HTMLDivElement

    constructor(uuid: string) {
        super()

        this.uuid = uuid
        this.currentChart = 0
        this.preferenceCache = comparisonCache.getPreferences(this.uuid)

        this.isUpdateDisable = true
        this.loader = new DataLoader(async (force) => {
            this.preferenceData = <ComparisonPreferenceModel>await this.preferenceCache.get(force)
            this.baseUuid = this.preferenceData.base_experiment
            this.status = await CACHE.getRunStatus(this.uuid).get(force)
            this.currentSeries = toPointValues((await metricsCache.getAnalysis(this.uuid).get(force)).series)
            this.baseSeries = toPointValues((await metricsCache.getAnalysis(this.baseUuid).get(force)).series)

            this.run = await CACHE.getRun(this.uuid).get(force)

            this.runHeaderCard = new RunHeaderCard({
                uuid: this.uuid,
                width: this.actualWidth / 2,
                showRank: false
            })

            if (!!this.baseUuid) {
                await this.updateBaseRun(force)
            } else {
                this.missingBaseExperiment = true
            }
        })

        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))
        this.deleteButton = new DeleteButton({onButtonClick: this.onDelete, parent: this.constructor.name})

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
        setTitle({section: 'ComparisonPreferenceModel'})
        clearChildElements(this.elem)
        $(this.elem, $ => {
            this.runPickerElem = $('div')
            $('div', '.page',
                {style: {width: `${this.actualWidth}px`}},
                $ => {
                    $('div', $ => {
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
                        this.toggleButtonContainer = $('div', '.button-row')
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

            this.content = new ViewWrapper({
                updatePreferences: this.updatePreferences,
                lineChartContainer: this.lineChartContainer,
                sparkLinesContainer: this.sparkLinesContainer,
                saveButtonContainer: this.saveButtonContainer,
                toggleButtonContainer: this.toggleButtonContainer,
                actualWidth: this.actualWidth,
                isUpdateDisable: this.isUpdateDisable,
                onRequestAllMetrics: this.requestAllMetrics.bind(this)
            })

            this.calcPreferences()
            this.renderHeaders()
            this.content.render()
            this.renderButtons()
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

    updatePreferences = (data?: ViewWrapperData, saveData: boolean = true) => {
        if (data) {
            this.currentPlotIdx = data.plotIdx
            this.basePlotIdx = data.basePlotIdx
            this.currentChart = data.currentChart
            this.focusSmoothed = data.focusSmoothed
            this.stepRange = data.stepRange
        }

        this.preferenceData.series_preferences = this.currentPlotIdx
        this.preferenceData.base_series_preferences = this.basePlotIdx
        this.preferenceData.chart_type = this.currentChart
        this.preferenceData.step_range = this.stepRange
        this.preferenceData.focus_smoothed = this.focusSmoothed

        if (!saveData) {
            return
        }

        this.preferenceCache.setPreference(this.preferenceData).then()

        this.isUpdateDisable = true
        this.content.renderSaveButton()
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

    private onDelete = () => {
        this.preferenceData.base_experiment = ''
        this.preferenceData.base_series_preferences = []
        this.baseSeries = undefined
        this.basePlotIdx = []
        this.baseUuid = ''
        this.baseRun = undefined

        this.updatePreferences()

        this.renderHeaders()
        this.content.render()
        this.renderButtons()
    }

    private onEditClick = () => {
        clearChildElements(this.runPickerElem)
        this.runPickerElem.classList.add("fullscreen-cover")
        document.body.classList.add("stop-scroll")
        this.runPickerElem.append(new RunsPickerView({
            title: 'Select run for comparison',
            excludedRuns: new Set<string>([this.run.run_uuid]),
            onPicked: async run => {
                if (this.preferenceData.base_experiment !== run.run_uuid) {
                    this.preferenceData.base_experiment = run.run_uuid
                    this.preferenceData.base_series_preferences = []
                    this.preferenceData.series_preferences = []
                    this.preferenceData.is_base_distributed = run.world_size != 0
                    this.baseUuid = run.run_uuid
                    this.basePlotIdx = []
                    this.currentPlotIdx = []
                    this.stepRange = [-1, -1]

                    this.isUpdateDisable = true

                    await this.updateBaseRun(false)

                    this.calcPreferences()

                    this.renderHeaders()
                    this.content.render()
                    this.renderButtons()
                }
                this.runPickerElem.classList.remove("fullscreen-cover")
                document.body.classList.remove("stop-scroll")
                clearChildElements(this.runPickerElem)
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
            new EditButton({
                onButtonClick: this.onEditClick,
                parent: this.constructor.name
            }).render($)
        })
    }

    private async updateBaseRun(force: boolean) {
        this.baseAnalysisCache = comparisonCache.getAnalysis(this.baseUuid)
        this.baseAnalysisCache.setCurrentUUID(this.uuid)
        this.baseRun = await CACHE.getRun(this.baseUuid).get(force)
        try {
            this.baseSeries = toPointValues((await this.baseAnalysisCache.get(force)).series)
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

    private requestAllMetrics() {  // TODO fix this to et only what is wanted
        metricsCache.getAnalysis(this.uuid).setMetricData(this.currentPlotIdx)
        metricsCache.getAnalysis(this.baseUuid).setMetricData(this.basePlotIdx)
        this.content.setLoading(true)
        this.isUpdateDisable = false
        this.loader.load(true).then(() => {
            this.calcPreferences()
            this.content.render()
            this.renderButtons()
        })
    }

    private calcPreferences() {
        if (this.isUpdateDisable) {
            this.currentChart = this.preferenceData.chart_type
            this.stepRange = this.preferenceData.step_range
            this.focusSmoothed = this.preferenceData.focus_smoothed
            this.basePlotIdx = [...this.preferenceData.series_preferences]
            this.currentPlotIdx = [...this.preferenceData.base_series_preferences]
        }

        this.content.updateData({
            series: this.currentSeries,
            baseSeries: this.baseSeries,
            basePlotIdx: this.basePlotIdx,
            plotIdx: this.currentPlotIdx,
            currentChart: this.currentChart,
            focusSmoothed: this.focusSmoothed,
            stepRange: this.stepRange
        })
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
