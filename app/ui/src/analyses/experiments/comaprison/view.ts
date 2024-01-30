import {Run, SeriesModel} from "../../../models/run"
import CACHE, {AnalysisDataCache, AnalysisPreferenceCache} from "../../../cache/cache"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {Status} from "../../../models/status"
import {DataLoader} from "../../../components/loader"
import {ROUTER, SCREEN} from "../../../app"
import {BackButton, DeleteButton, EditButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import {ComparisonPreferenceModel} from "../../../models/preferences"
import {fillPlotPreferences, toPointValues} from "../../../components/charts/utils"
import mix_panel from "../../../mix_panel"
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

// TODO missing base experiment
class ComparisonView extends ScreenView implements MetricDataStore {
    private readonly uuid: string
    private baseUuid: string

    private preferenceData: ComparisonPreferenceModel
    private status: Status

    series: SeriesModel[]
    baseSeries?: SeriesModel[]
    plotIdx: number[]
    basePlotIdx?: number[]
    chartType: number
    focusSmoothed: boolean
    stepRange: number[]

    preservePreferences: boolean

    private elem: HTMLDivElement
    private runHeaderCard: RunHeaderCard
    private lineChartContainer: HTMLDivElement
    private sparkLinesContainer: HTMLDivElement
    private buttonContainer: HTMLDivElement
    private optionRowContainer: HTMLDivElement
    private runPickerElem: HTMLDivElement
    private headerContainer: HTMLDivElement
    private saveButtonContainer: HTMLDivElement

    private actualWidth: number
    private refresh: AwesomeRefreshButton

    private loader: DataLoader
    private content: ViewWrapper

    private preferenceCache: AnalysisPreferenceCache
    private run: Run
    private baseRun: Run
    private baseAnalysisCache: AnalysisDataCache
    private missingBaseExperiment: Boolean
    private deleteButton: DeleteButton


    constructor(uuid: string) {
        super()

        this.uuid = uuid
        this.chartType = 0
        this.preferenceCache = comparisonCache.getPreferences(this.uuid)

        this.loader = new DataLoader(async (force) => {
            this.status = await CACHE.getRunStatus(this.uuid).get(force)
            this.run = await CACHE.getRun(this.uuid).get(force)

            this.series = toPointValues((await metricsCache.getAnalysis(this.uuid).get(force)).series)

            this.preferenceData = <ComparisonPreferenceModel>await this.preferenceCache.get(force)
            this.baseUuid = this.preferenceData.base_experiment
            if (this.baseUuid != '') {
                this.baseSeries = toPointValues((await metricsCache.getAnalysis(this.baseUuid).get(force)).series)
                this.preferenceData.base_series_preferences = fillPlotPreferences(this.baseSeries)
            }
            this.preferenceData.series_preferences = fillPlotPreferences(this.series)


            if (!!this.baseUuid) {
                await this.updateBaseRun(force)
            } else {
                this.missingBaseExperiment = true
            }

            if(!this.preservePreferences) {
                this.chartType = this.preferenceData.chart_type
                this.stepRange = this.preferenceData.step_range
                this.focusSmoothed = this.preferenceData.focus_smoothed
                this.plotIdx = this.preferenceData.series_preferences
                this.basePlotIdx = this.preferenceData.base_series_preferences
            }
        })

        this.runHeaderCard = new RunHeaderCard({
            uuid: this.uuid,
            width: this.actualWidth / 2,
            showRank: false
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
        setTitle({section: 'Comparison'})
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
                        this.optionRowContainer = $('div', '.button-row')
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
                dataStore: this,
                lineChartContainer: this.lineChartContainer,
                sparkLinesContainer: this.sparkLinesContainer,
                saveButtonContainer: this.saveButtonContainer,
                optionRowContainer: this.optionRowContainer,
                actualWidth: this.actualWidth,
                requestMissingMetrics: this.requestMissingMetrics.bind(this),
                savePreferences: this.savePreferences.bind(this)
            })

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

    private savePreferences = () => {
        this.preferenceData.series_preferences = this.plotIdx
        this.preferenceData.base_series_preferences = this.basePlotIdx
        this.preferenceData.chart_type = this.chartType
        this.preferenceData.step_range = this.stepRange
        this.preferenceData.focus_smoothed = this.focusSmoothed

        this.preferenceCache.setPreference(this.preferenceData).then()

        this.preservePreferences = false
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

        this.savePreferences()

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
                    this.preferenceData.is_base_distributed = run.world_size != 0
                    this.baseUuid = run.run_uuid
                    this.basePlotIdx = []
                    this.plotIdx = []
                    this.stepRange = [-1, -1]

                    this.preservePreferences = false

                    await this.updateBaseRun(false)

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

    private async requestMissingMetrics() {
        metricsCache.getAnalysis(this.uuid).setMetricData(this.plotIdx)
        this.series = toPointValues((await metricsCache.getAnalysis(this.uuid).get(true)).series)

        if (this.baseUuid !== '') {
            metricsCache.getAnalysis(this.baseUuid).setMetricData(this.basePlotIdx)
            this.baseSeries = toPointValues((await metricsCache.getAnalysis(this.baseUuid).get(true)).series)
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
