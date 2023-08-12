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
import {BackButton, ToggleButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import {CompareLineChart} from "../../../components/charts/compare_lines/chart"
import {ErrorMessage} from "../../../components/error_message"
import {CompareSparkLines} from "../../../components/charts/compare_spark_lines/chart"
import {NetworkError} from "../../../network"

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
    private baseRunHeaderCard: RunHeaderCard
    private headerContainer: HTMLDivElement
    private lineChartContainer: HTMLDivElement
    private sparkLineContainer: HTMLDivElement
    private missingBaseExperiment: Boolean
    private sparkLines: CompareSparkLines
    private toggleButtonContainer: HTMLDivElement

    constructor(uuid: string) {
        super()

        this.currentUuid = uuid
        this.runCache = CACHE.getRun(this.currentUuid)
        this.statusCache = CACHE.getRunStatus(this.currentUuid)
        this.preferenceCache = comparisonCache.getPreferences(this.currentUuid)
        this.currentAnalysisCache = comparisonCache.getAnalysis(this.currentUuid)

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
        this.preferenceData.chart_type ^= 1

        this.renderLineChart()
    }

    private toggleCurrentChart = (idx: number) => {
        if (this.preferenceData.series_preferences[idx] == -1) {
            this.preferenceData.series_preferences[idx] = Math.max(...this.preferenceData.series_preferences) + 1
        } else {
            this.preferenceData.series_preferences[idx] = -1
        }

        this.renderSparkLineChart()
        this.renderLineChart()
    }

    private toggleBaseChart = (idx: number) => {
        if (this.preferenceData.base_series_preferences[idx] == -1) {
            this.preferenceData.base_series_preferences[idx] = Math.max(...this.preferenceData.base_series_preferences) + 1
        } else {
            this.preferenceData.base_series_preferences[idx] = -1
        }

        this.renderSparkLineChart()
        this.renderLineChart()
    }

    private renderHeaders() {
        clearChildElements(this.headerContainer)
        $(this.headerContainer,  $=> {
            $('div', '',
                async $ => {
                    await this.runHeaderCard.render($)
                })
            $('span', '.fas.fa-exchange-alt', '')
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
                    isToggled: this.preferenceData.chart_type > 0,
                    parent: this.constructor.name
                })
                    .render($)
            })
        }
    }

    private renderLineChart() {
        clearChildElements(this.lineChartContainer)

        if (!!this.baseSeries) {
            $(this.lineChartContainer, $ => {
                new CompareLineChart({
                    series: this.currentSeries,
                    baseSeries: this.baseSeries,
                    currentPlotIdx: [...(this.preferenceData.series_preferences ?? [])],
                    basePlotIdx: [...(this.preferenceData.base_series_preferences ?? [])],
                    width: this.actualWidth,
                    chartType: getChartType(this.preferenceData.chart_type),
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
                    currentPlotIdx: [...(this.preferenceData.series_preferences ?? [])],
                    basePlotIdx: [...(this.preferenceData.base_series_preferences ?? [])],
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
            $('div', '.page', {style: {width: `${this.actualWidth}px`}}, $ => {
                $('div', '.nav-container', $ => {
                    this.backButton.render($)
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
            this.renderHeaders()
            this.renderSparkLineChart() // has to run before render line chart as it uses the spark line component
            this.renderLineChart()
            this.renderToggleButtons()
            setTitle({section: 'Comparison', item: this.run.name})
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
