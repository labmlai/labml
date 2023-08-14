import {ROUTER, SCREEN} from "../../../app"
import {ScreenView} from "../../../screen_view"
import {ViewHandler} from "../../types"
import CACHE, {AnalysisDataCache, AnalysisPreferenceCache, RunCache, RunStatusCache} from "../../../cache/cache"
import comparisonCache from "./cache"
import {DataLoader} from "../../../components/loader"
import {ComparisonPreferenceModel} from "../../../models/preferences"
import {Status} from "../../../models/status"
import {Run, SeriesModel} from "../../../models/run"
import {getChartType, toPointValues} from "../../../components/charts/utils"
import mix_panel from "../../../mix_panel"
import {clearChildElements, setTitle} from "../../../utils/document"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {BackButton, EditButton, SaveButton, ToggleButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import {CompareLineChart} from "../../../components/charts/compare_lines/chart"
import {ErrorMessage} from "../../../components/error_message"
import {CompareSparkLines} from "../../../components/charts/compare_spark_lines/chart"
import {NetworkError} from "../../../network"
import {RunsPickerView} from "../../../views/run_picker_view"

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
    private baseRun: Run
    private currentSeries: SeriesModel[]
    private baseSeries: SeriesModel[]
    private actualWidth: number
    private backButton: BackButton
    private runHeaderCard: RunHeaderCard
    private baseRunHeaderCard: RunHeaderCard
    private headerContainer: HTMLDivElement
    private lineChartContainer: HTMLDivElement
    private sparkLineContainer: HTMLDivElement
    private missingBaseExperiment: Boolean
    private sparkLines: CompareSparkLines
    private toggleButtonContainer: HTMLDivElement
    private currentPlotIdx: number[]
    private basePlotIdx: number[]
    private currentChart: number // log or linear
    private runPickerElem: HTMLDivElement
    private saveButton: SaveButton
    private buttonContainer: HTMLDivElement
    private shouldPreservePreferences: boolean
    private isUpdateDisabled: boolean

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
            this.status = await this.statusCache.get(force)
            this.run = await this.runCache.get()
            this.currentSeries = toPointValues((await this.currentAnalysisCache.get(force)).series)
            this.runHeaderCard = new RunHeaderCard({
                uuid: this.currentUuid,
                width: this.actualWidth/2
            })
            if (!!this.baseUuid) {
                this.baseRunHeaderCard = new RunHeaderCard({
                    uuid: this.baseUuid,
                    width: this.actualWidth/2
                })
                this.baseRun = await CACHE.getRun(this.baseUuid).get()
                this.baseAnalysisCache = comparisonCache.getAnalysis(this.baseUuid)
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
            } else {
                this.baseAnalysisCache = undefined
                this.baseSeries = undefined
                this.missingBaseExperiment = true
            }
        })

        mix_panel.track('Analysis View', {uuid: this.currentUuid, analysis: this.constructor.name})

        this.backButton = new BackButton({text: 'Run', parent: this.constructor.name})
        this.saveButton = new SaveButton({onButtonClick: this.updatePreferences, parent: this.constructor.name})
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

    private onChangeScale() {
        this.shouldPreservePreferences = true
        this.isUpdateDisabled = false

        this.currentChart ^= 1

        this.renderLineChart()
        this.renderButtons()
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

        this.renderSparkLineChart()
        this.renderLineChart()
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

        this.renderSparkLineChart()
        this.renderLineChart()
        this.renderButtons()
    }

    private calcPreferences() {
        if (this.shouldPreservePreferences)
            return

        this.currentChart = this.preferenceData.chart_type

        let currentAnalysisPreferences = this.preferenceData.series_preferences
        if (currentAnalysisPreferences && currentAnalysisPreferences.length > 0) {
            this.currentPlotIdx = [...currentAnalysisPreferences]
        } else if (this.currentSeries) {
            let res: number[] = []
            for (let i = 0; i < this.currentSeries.length; i++) {
                res.push(i)
            }
            this.currentPlotIdx = res
        }

        let baseAnalysisPreferences = this.preferenceData.base_series_preferences
        if (baseAnalysisPreferences && baseAnalysisPreferences.length > 0) {
            this.basePlotIdx = [...baseAnalysisPreferences]
        } else if (this.baseSeries) {
            let res: number[] = []
            for (let i = 0; i < this.baseSeries.length; i++) {
                res.push(i)
            }
            this.basePlotIdx = res
        }
    }

    private updatePreferences = () => {
        this.preferenceData.series_preferences = this.currentPlotIdx
        this.preferenceData.base_series_preferences = this.basePlotIdx
        this.preferenceData.chart_type = this.currentChart
        this.preferenceCache.setPreference(this.preferenceData).then()

        this.shouldPreservePreferences = false
        this.isUpdateDisabled = true

        this.renderButtons()
    }

    private onEditClick = () => {
        clearChildElements(this.runPickerElem)
        this.runPickerElem.classList.add("fullscreen-cover")
        this.runPickerElem.append(new RunsPickerView({
                title: 'Select run for comparison',
                excludedRuns: new Set<string>([this.run.run_uuid]),
                onPicked: run => {
                    this.runPickerElem.classList.remove("fullscreen-cover")
                    clearChildElements(this.runPickerElem)
                }, onCancel: () => {
                    this.runPickerElem.classList.remove("fullscreen-cover")
                    clearChildElements(this.runPickerElem)
                }
            })
                .render())
    }

    private renderHeaders() {
        clearChildElements(this.headerContainer)
        $(this.headerContainer,  $=> {
            $('div', '',
                async $ => {
                    await this.runHeaderCard.render($)
                })
            $('span', '.compared-with', $ => {
                $('span', '.text', 'Compared With')
                new EditButton({
                    onButtonClick: this.onEditClick,
                    parent: this.constructor.name
                }).render($)
            })
            $('div', '',
                async $ => {
                    await this.baseRunHeaderCard.render($)
                })
        })
    }

    private renderToggleButtons() {
        clearChildElements(this.toggleButtonContainer)
        if (!!this.baseSeries) {
            $(this.toggleButtonContainer, $ => {
                new ToggleButton({
                    onButtonClick: this.onChangeScale.bind(this),
                    text: 'Log',
                    isToggled: this.currentChart > 0,
                    parent: this.constructor.name
                })
                    .render($)
            })
        }
    }

    private renderButtons() {
        clearChildElements(this.buttonContainer)
        this.saveButton.disabled = this.isUpdateDisabled
        $(this.buttonContainer, $ => {
            this.saveButton.render($)
        })
    }

    private renderLineChart() {
        clearChildElements(this.lineChartContainer)

        if (!!this.baseSeries) {
            $(this.lineChartContainer, $ => {
                new CompareLineChart({
                    series: this.currentSeries,
                    baseSeries: this.baseSeries,
                    currentPlotIdx: this.currentPlotIdx,
                    basePlotIdx: this.basePlotIdx,
                    width: this.actualWidth,
                    chartType: getChartType(this.currentChart),
                    isDivergent: true,
                    isCursorMoveOpt: true,
                    onCursorMove: [this.sparkLines.changeCursorValues]
                }).render($)
            })
        } else if (this.missingBaseExperiment) {
            (new ErrorMessage('Base Experiment Not Found')).render(this.lineChartContainer)
        }
    }

    private renderSparkLineChart() {
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
                })
                this.loader.render($)
                this.headerContainer = $('div', '.compare-header')
                this.toggleButtonContainer = $('div')
                $('h2', '.header.text-center', 'Comparison')
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
            this.renderSparkLineChart() // has to run before render line chart as it uses the spark line component
            this.renderLineChart()
            this.renderToggleButtons()
            this.renderButtons()
        } catch (e) {
            // todo handle network error
            console.log(e)
        } finally {
            // todo refresh
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
