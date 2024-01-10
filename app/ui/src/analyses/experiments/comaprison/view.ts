import {ROUTER, SCREEN} from "../../../app"
import {ScreenView} from "../../../screen_view"
import {ViewHandler} from "../../types"
import CACHE, {AnalysisDataCache, AnalysisPreferenceCache, RunCache, RunStatusCache} from "../../../cache/cache"
import comparisonCache from "./cache"
import {DataLoader} from "../../../components/loader"
import {ComparisonPreferenceModel} from "../../../models/preferences"
import {Status} from "../../../models/status"
import {Run, SeriesModel} from "../../../models/run"
import {defaultSeriesToPlot, getChartType, toPointValues} from "../../../components/charts/utils"
import mix_panel from "../../../mix_panel"
import {clearChildElements, setTitle} from "../../../utils/document"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {BackButton, DeleteButton, EditButton, SaveButton, ToggleButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import {LineChart} from "../../../components/charts/compare_lines/chart"
import {ErrorMessage} from "../../../components/error_message"
import {CompareSparkLines} from "../../../components/charts/compare_spark_lines/chart"
import {NetworkError} from "../../../network"
import {RunsPickerView} from "../../../views/run_picker_view"
import {AwesomeRefreshButton} from "../../../components/refresh_button"
import {DEBUG} from "../../../env"
import {handleNetworkErrorInplace} from "../../../utils/redirect"
import {NumericRangeField} from "../../../components/input/numeric_range_field"
import {DropDownMenu} from "../../../components/dropdown_button"


class ComparisonView extends ScreenView {
    private elem: HTMLDivElement
    private readonly currentUuid: string
    private baseUuid: string
    private preferenceCache: AnalysisPreferenceCache
    private runCache: RunCache
    private statusCache: RunStatusCache
    private currentAnalysisCache: AnalysisDataCache
    private baseAnalysisCache: AnalysisDataCache
    private preferenceData: ComparisonPreferenceModel
    private loader: DataLoader
    private status: Status
    private run: Run
    private currentSeries: SeriesModel[]
    private baseSeries: SeriesModel[]
    private actualWidth: number
    private backButton: BackButton
    private runHeaderCard: RunHeaderCard
    private headerContainer: HTMLDivElement
    private lineChartContainer: HTMLDivElement
    private sparkLineContainer: HTMLDivElement
    private missingBaseExperiment: Boolean
    private sparkLines: CompareSparkLines
    private optionContainer: HTMLDivElement
    private currentPlotIdx: number[]
    private basePlotIdx: number[]
    private currentChart: number // log or linear
    private runPickerElem: HTMLDivElement
    private saveButton: SaveButton
    private buttonContainer: HTMLDivElement
    private shouldPreservePreferences: boolean
    private isUpdateDisabled: boolean
    private refresh: AwesomeRefreshButton
    private baseRun: Run
    private deleteButton: DeleteButton
    private stepRange: number[]
    private stepRangeField: NumericRangeField
    private chartTypeButton: ToggleButton
    private stepDropDown: DropDownMenu
    private focusSmoothed: boolean

    constructor(uuid: string) {
        super()

        this.currentUuid = uuid
        this.runCache = CACHE.getRun(this.currentUuid)
        this.statusCache = CACHE.getRunStatus(this.currentUuid)
        this.preferenceCache = comparisonCache.getPreferences(this.currentUuid)
        this.currentAnalysisCache = comparisonCache.getAnalysis(this.currentUuid)
        this.shouldPreservePreferences = false
        this.isUpdateDisabled = true

        this.loader = new DataLoader(async (force) => {
            this.preferenceData = <ComparisonPreferenceModel>await this.preferenceCache.get(force)
            this.baseUuid = this.preferenceData.base_experiment
            this.stepRange = this.preferenceData.step_range
            this.currentChart = this.preferenceData.chart_type
            this.focusSmoothed = this.preferenceData.focus_smoothed
            this.status = await this.statusCache.get(force)
            this.run = await this.runCache.get()
            this.currentSeries = toPointValues((await this.currentAnalysisCache.get(force)).series)
            this.runHeaderCard = new RunHeaderCard({
                uuid: this.currentUuid,
                width: this.actualWidth/2,
                showRank: false
            })
            if (!!this.baseUuid) {
                await this.updateBaseRun(force)
            } else {
                this.baseAnalysisCache = undefined
                this.baseSeries = undefined
            }
        })

        mix_panel.track('Analysis View', {uuid: this.currentUuid, analysis: this.constructor.name})

        this.backButton = new BackButton({text: 'Run', parent: this.constructor.name})
        this.saveButton = new SaveButton({onButtonClick: this.updatePreferences, parent: this.constructor.name})
        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))
        this.deleteButton = new DeleteButton({onButtonClick: this.onDelete, parent: this.constructor.name})
        this.stepRangeField = new NumericRangeField({
            max: 0, min: 0,
            onClick: this.onChangeStepRange.bind(this),
            buttonLabel: "Filter Steps"
        })
        this.chartTypeButton = new ToggleButton({
            onButtonClick: this.onChangeScale.bind(this),
            text: 'Log',
            isToggled: this.currentChart > 0,
            parent: this.constructor.name
        })
    }

    get requiresAuth(): boolean {
        return  false
    }

    onResize(width: number) {
        super.onResize(width)

        this.actualWidth = Math.min(800, width)

        if (this.elem) {
            this._render().then()
        }
    }

    private onChangeStepRange(min: number, max: number) {
        this.shouldPreservePreferences = true
        this.isUpdateDisabled = false

        this.stepRange = [min, max]

        this.renderCharts()
        this.renderButtons()
        this.renderOptionRow()
    }

    private onChangeSmoothFocus() {
        this.shouldPreservePreferences = true
        this.isUpdateDisabled = false

        this.focusSmoothed = !this.focusSmoothed

        this.renderCharts()
        this.renderButtons()
    }

    private onChangeScale() {
        this.shouldPreservePreferences = true
        this.isUpdateDisabled = false

        this.currentChart ^= 1

        this.renderCharts()
        this.renderButtons()
    }

    private async updateBaseRun(force: boolean) {
        this.baseAnalysisCache = comparisonCache.getAnalysis(this.baseUuid)
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

    private toggleCurrentChart = (idx: number) => {
        this.shouldPreservePreferences = true
        this.isUpdateDisabled = false

        if (this.currentPlotIdx[idx] >= 0) {
            this.currentPlotIdx[idx] = -1
        } else {
            this.currentPlotIdx[idx] = Math.max(...this.currentPlotIdx) + 1
        }

        if (this.currentPlotIdx.length > 1) {
            this.currentPlotIdx = new Array<number>(...this.currentPlotIdx)
        }

        this.renderCharts()
        this.renderButtons()
    }

    private toggleBaseChart = (idx: number) => {
        this.shouldPreservePreferences = true
        this.isUpdateDisabled = false

        if (this.basePlotIdx[idx] >= 0) {
            this.basePlotIdx[idx] = -1
        } else {
            this.basePlotIdx[idx] = Math.max(...this.basePlotIdx) + 1
        }

        if (this.basePlotIdx.length > 1) {
            this.basePlotIdx = new Array<number>(...this.basePlotIdx)
        }

        this.renderCharts()
        this.renderButtons()
    }

    private onFilterDropdownClick = (id: string) => {
        let minStep: number, maxStep: number
        if (id === 'base') {
            minStep = Math.min(...this.baseSeries.map(s => s.series[0].step))
            maxStep = Math.max(...this.baseSeries.map(s => s.series[s.series.length - 1].step))
        } else if (id === 'current') {
            minStep = Math.min(...this.currentSeries.map(s => s.series[0].step))
            maxStep = Math.max(...this.currentSeries.map(s => s.series[s.series.length - 1].step))
        } else if (id === 'all') {
            minStep = -1
            maxStep = -1
        }

        this.onChangeStepRange(minStep, maxStep)
    }

    private calcPreferences() {
        if (this.shouldPreservePreferences)
            return

        this.currentChart = this.preferenceData.chart_type
        this.stepRange = this.preferenceData.step_range
        this.focusSmoothed = this.preferenceData.focus_smoothed

        let currentAnalysisPreferences = this.preferenceData.series_preferences
        if (currentAnalysisPreferences && currentAnalysisPreferences.length > 0) {
            this.currentPlotIdx = [...currentAnalysisPreferences]
        } else if (this.currentSeries) {
            this.currentPlotIdx = defaultSeriesToPlot(this.currentSeries)
        }

        let baseAnalysisPreferences = this.preferenceData.base_series_preferences
        if (baseAnalysisPreferences && baseAnalysisPreferences.length > 0) {
            this.basePlotIdx = [...baseAnalysisPreferences]
        } else if (this.baseSeries) {
            this.basePlotIdx = defaultSeriesToPlot(this.baseSeries)
        }
    }

    private updatePreferences = () => {
        this.preferenceData.series_preferences = this.currentPlotIdx
        this.preferenceData.base_series_preferences = this.basePlotIdx
        this.preferenceData.chart_type = this.currentChart
        this.preferenceData.step_range = this.stepRange
        this.preferenceData.focus_smoothed = this.focusSmoothed
        this.preferenceCache.setPreference(this.preferenceData).then()

        this.shouldPreservePreferences = false
        this.isUpdateDisabled = true

        this.renderButtons()
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
        this.renderCharts()
        this.renderOptionRow()
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
                        this.isUpdateDisabled = true
                        this.shouldPreservePreferences = false

                        this.preferenceData.base_experiment = run.run_uuid
                        this.preferenceData.base_series_preferences = []
                        this.preferenceData.series_preferences = []
                        this.preferenceData.is_base_distributed = run.world_size != 0
                        this.baseUuid = run.run_uuid
                        this.basePlotIdx = []
                        this.currentPlotIdx = []
                        this.stepRange = [-1, -1]

                        await this.updateBaseRun(false)

                        this.updatePreferences()
                        this.calcPreferences()

                        this.renderHeaders()
                        this.renderCharts()
                        this.renderOptionRow()
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

    onVisibilityChange() {
        this.refresh.changeVisibility(!document.hidden)
    }

    async onRefresh() {
        try {
            await this.loader.load(true)

            this.calcPreferences()
            this.renderCharts()
        } catch (e) {

        } finally {
            if (this.status && !this.status.isRunning) {
                this.refresh.stop()
            }
            await this.runHeaderCard.refresh()
        }
    }

    private renderHeaders() {
        clearChildElements(this.headerContainer)
        $(this.headerContainer,  $=> {
            this.runHeaderCard.render($).then()
            $('span', '.compared-with', $ => {
                $('span', '.sub', 'Compared With ')
                if (this.baseRun == null) {
                    $('span', '.title', 'No run selected')
                } else {
                    $('a', '.title.clickable', `${this.baseRun.name} `, {on: {click: () => {
                                window.open(`/run/${this.baseRun.run_uuid}`, '_blank')
                            }}})
                }
            })
        })
    }

    private renderOptionRow() {
        clearChildElements(this.optionContainer)
        if (this.baseSeries == null)
            return
        this.chartTypeButton.isToggled = this.currentChart > 0
        this.stepRangeField.setRange(this.stepRange[0], this.stepRange[1])
        this.stepDropDown = new DropDownMenu({
            items: [{id: 'all', title: 'All'},
                {id: 'base', title: this.baseRun.name},
                {id: 'current', title: this.run.name}],
            onItemSelect: this.onFilterDropdownClick.bind(this),
            parent: this.constructor.name, title: ""
        })
        $(this.optionContainer, $ => {
            this.chartTypeButton.render($)
            new ToggleButton({
                onButtonClick: this.onChangeSmoothFocus.bind(this),
                text: 'Focus Smoothed',
                isToggled: this.focusSmoothed,
                parent: this.constructor.name
            })
                .render($)
            $('div.button-row', $ => {
                this.stepRangeField.render($)
                this.stepDropDown.render($)
            })
        })
    }

    private renderButtons() {
        clearChildElements(this.buttonContainer)
        this.saveButton.disabled = this.isUpdateDisabled
        this.deleteButton.disabled = !this.baseUuid
        $(this.buttonContainer, $ => {
            this.deleteButton.render($)
            this.saveButton.render($)
            new EditButton({
                    onButtonClick: this.onEditClick,
                    parent: this.constructor.name
                }).render($)
        })
    }

    private renderCharts() {
        clearChildElements(this.sparkLineContainer)
        $(this.sparkLineContainer, $=> {
            if (!!this.baseSeries) {
                this.sparkLines = new CompareSparkLines({
                    series: this.currentSeries,
                    baseSeries: this.baseSeries,
                    currentPlotIdx: this.currentPlotIdx,
                    basePlotIdx: this.basePlotIdx,
                    width: this.actualWidth,
                    isDivergent: true,
                    onCurrentSelect: this.toggleCurrentChart,
                    onBaseSelect: this.toggleBaseChart
                })
                this.sparkLines.render($)
            }
        })

        clearChildElements(this.lineChartContainer)
        if (!!this.baseSeries) {
            $(this.lineChartContainer, $ => {
                new LineChart({
                    series: this.currentSeries,
                    baseSeries: this.baseSeries,
                    currentPlotIndex: this.currentPlotIdx,
                    basePlotIdx: this.basePlotIdx,
                    width: this.actualWidth,
                    chartType: getChartType(this.currentChart),
                    isDivergent: true,
                    isCursorMoveOpt: true,
                    onCursorMove: [this.sparkLines.changeCursorValues],
                    stepRange: this.stepRange,
                    focusSmoothed: this.focusSmoothed
                }).render($)
            })
        } else if (this.missingBaseExperiment) {
            (new ErrorMessage('Base Experiment Not Found')).render(this.lineChartContainer)
        }
    }

    async _render(): Promise<void> {
        setTitle({section: 'Comparison'})
        clearChildElements(this.elem)
        $(this.elem, $ => {
            this.runPickerElem = $('div')
            $('div', '.page', {style: {width: `${this.actualWidth}px`}}, $ => {
                $('div', '.nav-container', $ => {
                    this.backButton.render($)
                    this.buttonContainer = $('div')
                    this.refresh.render($)
                })
                this.loader.render($)
                this.headerContainer = $('div', '.compare-header')
                this.optionContainer = $('div', '.button-row')
                if (this.baseRun != null) {
                    $('h2', '.header.text-center', 'Comparison')
                }
                $('div', '.detail-card', $ => {
                    this.lineChartContainer = $('div', '.fixed-chart')
                    this.sparkLineContainer = $('div')
                })
            })
        })
        try {
            await this.loader.load()

            setTitle({section: 'Comparison', item: this.run.name})
            this.calcPreferences()

            this.renderHeaders()
            this.renderCharts()
            this.renderOptionRow()
            this.renderButtons()
        } catch (e) {
            handleNetworkErrorInplace(e)
            if (DEBUG) {
                console.log(e)
            }
        } finally {
            if (this.status?.isRunning == true) {
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
