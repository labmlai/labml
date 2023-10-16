import {SeriesModel} from "../../../models/run"
import CACHE from "../../../cache/cache"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {Status} from "../../../models/status"
import {DataLoader} from "../../../components/loader"
import {ROUTER, SCREEN} from "../../../app"
import {BackButton, SaveButton, ToggleButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import {DistAnalysisPreferenceModel} from "../../../models/preferences"
import {LineChart} from "../../../components/charts/lines/chart"
import {SparkLines} from "../../../components/charts/spark_lines/chart"
import {getChartType, toPointValues} from "../../../components/charts/utils"
import mix_panel from "../../../mix_panel"
import {ViewHandler} from "../../types"
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {setTitle} from '../../../utils/document'
import {ScreenView} from '../../../screen_view'
import {NumericRangeField} from "../../../components/input/numeric_range_field"
import {DistMetricsAnalysisCache, DistMetricsPreferenceCache} from "./cache"

class DistributedMetricsView extends ScreenView {
    uuid: string

    series: SeriesModel[]
    preferenceData: DistAnalysisPreferenceModel
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
    private preferenceCache: DistMetricsPreferenceCache

    private singleSeriesLength: number

    constructor(uuid: string) {
        super()
        this.uuid = uuid
        this.currentChart = 0

        this.isUpdateDisable = true

        this.loader = new DataLoader(async (force) => {
            this.status = await CACHE.getRunStatus(this.uuid).get(force)

            this.preferenceData = {
                series_preferences: [],
                sub_series_preferences: {},
                chart_type: 0,
                step_range: [-1, -1],
                focus_smoothed: false
            }

            let run = await CACHE.getRun(this.uuid).get(false)
            let worldSize = run.world_size

            if (worldSize == 0)
                return

            this.series = []
            let metricCache = new DistMetricsAnalysisCache(this.uuid, CACHE.getRunStatus(this.uuid))

            let analysisData = await metricCache.get(force)
            analysisData.series.forEach((series, index) => {
                let s: SeriesModel[] = toPointValues(series)
                for (let item of s) {
                    item.name = `rank${index+1} ${item.name}`
                }
                this.series = this.series.concat(s)
                this.singleSeriesLength = series.length
            })

            this.preferenceCache = new DistMetricsPreferenceCache(this.uuid, worldSize, analysisData.series[0].length)
            this.preferenceData = await this.preferenceCache.get(force)
            console.log(this.preferenceData)
           // this.preferenceData.series_preferences = Array.from({ length: this.series.length }, (_, index) => index + 1)
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
                        this.toggleButtonContainer = $('div.button-row')
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
                this.plotIdx = [].concat(...analysisPreferences)
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


        let seriesPreferences: number[][] = []
        let _plotIdx = this.plotIdx.slice(0)
        while (_plotIdx.length > 0) {
            seriesPreferences.push(_plotIdx.splice(0, this.singleSeriesLength))
        }
        this.preferenceData.series_preferences = seriesPreferences
        this.preferenceData.chart_type = this.currentChart
        this.preferenceData.step_range = this.stepRange
        this.preferenceData.focus_smoothed = this.focusSmoothed
        this.preferenceCache.setPreference(this.preferenceData).then()
        console.log(this.preferenceData)

        this.isUpdateDisable = true
        this.content.renderSaveButton()
    }
}

interface ViewContentOpt {
    updatePreferences: (data: ViewContentData) => void
    lineChartContainer: HTMLDivElement
    sparkLinesContainer: HTMLDivElement
    saveButtonContainer: WeyaElement
    toggleButtonContainer: WeyaElement
    actualWidth: number

    isUpdateDisable: boolean
}

export interface ViewContentData {
    series?: SeriesModel[]
    plotIdx?: number[]
    currentChart?: number
    focusSmoothed?: boolean
    stepRange?: number[]
}

export class DistributedViewContent {
    private sparkLines: SparkLines
    private readonly lineChartContainer: HTMLDivElement
    private readonly sparkLinesContainer: HTMLDivElement
    private readonly saveButtonContainer: WeyaElement
    private readonly toggleButtonContainer: WeyaElement
    private isUpdateDisable: boolean
    private actualWidth: number

    private readonly stepRangeField: NumericRangeField
    private readonly saveButton: SaveButton

    private series: SeriesModel[]

    private plotIdx: number[] = []
    private currentChart: number
    private focusSmoothed: boolean
    private stepRange: number[]

    constructor(opt: ViewContentOpt) {
        this.lineChartContainer = opt.lineChartContainer
        this.sparkLinesContainer = opt.sparkLinesContainer
        this.saveButtonContainer = opt.saveButtonContainer
        this.toggleButtonContainer = opt.toggleButtonContainer
        this.actualWidth = opt.actualWidth

        this.stepRangeField = new NumericRangeField({
            max: 0, min: 0,
            onClick: this.onChangeStepRange.bind(this),
            buttonLabel: "Filter Steps"
        })

        this.saveButton = new SaveButton({onButtonClick: () => {
            opt.updatePreferences({
                series: this.series,
                plotIdx: this.plotIdx,
                currentChart: this.currentChart,
                focusSmoothed: this.focusSmoothed,
                stepRange: this.stepRange
            })
            }, parent: this.constructor.name})
    }

    public updateData(data: ViewContentData) {
        this.series = data.series != null ? data.series : this.series
        this.plotIdx = data.plotIdx != null ? data.plotIdx : this.plotIdx
        this.currentChart = data.currentChart != null ? data.currentChart : this.currentChart
        this.focusSmoothed = data.focusSmoothed != null ? data.focusSmoothed : this.focusSmoothed
        this.stepRange = data.stepRange != null ? data.stepRange : this.stepRange

        this.stepRangeField.setRange(this.stepRange[0], this.stepRange[1])
    }

    public renderCharts() {
        this.renderSparkLines()
        this.renderLineChart()
    }

    public render() {
        this.renderCharts()
        this.renderSaveButton()
        this.renderToggleButton()
    }

    public renderSaveButton() {
        this.saveButton.disabled = this.isUpdateDisable
        this.saveButtonContainer.innerHTML = ''
        $(this.saveButtonContainer, $ => {
            this.saveButton.render($)
        })
    }

    private onChangeStepRange(min: number, max: number) {
        this.isUpdateDisable = false

        this.stepRange = [min, max]

        this.renderLineChart()
        this.renderSaveButton()
        this.renderToggleButton()
    }

    private renderToggleButton() {
        this.toggleButtonContainer.innerHTML = ''
        $(this.toggleButtonContainer, $ => {
            new ToggleButton({
                onButtonClick: this.onChangeScale,
                text: 'Log',
                isToggled: this.currentChart > 0,
                parent: this.constructor.name
            }).render($)
            new ToggleButton({
                onButtonClick: this.onChangeSmoothFocus,
                text: 'Focus Smoothed',
                isToggled: this.focusSmoothed,
                parent: this.constructor.name
            })
                .render($)
            this.stepRangeField.render($)
        })
    }

    private renderLineChart() {
        this.lineChartContainer.innerHTML = ''
        $(this.lineChartContainer, $ => {
            new LineChart({
                series: this.series,
                width: this.actualWidth,
                plotIdx: this.plotIdx,
                chartType: getChartType(this.currentChart),
                onCursorMove: [this.sparkLines.changeCursorValues],
                isCursorMoveOpt: true,
                isDivergent: true,
                stepRange: this.stepRange,
                focusSmoothed: this.focusSmoothed,
                isDistributed: true
            }).render($)
        })
    }

    private renderSparkLines() {
        this.sparkLinesContainer.innerHTML = ''
        $(this.sparkLinesContainer, $ => {
            this.sparkLines = new SparkLines({
                series: this.series,
                plotIdx: this.plotIdx,
                width: this.actualWidth,
                onSelect: this.toggleChart,
                isDivergent: true,
                isDistributed: true
            })
            this.sparkLines.render($)
        })
    }

    private toggleChart = (idx: number) => {
        this.isUpdateDisable = false

        if (this.plotIdx[idx] >= 0) {
            this.plotIdx[idx] = -1
        } else {
            this.plotIdx[idx] = Math.max(...this.plotIdx) + 1
        }

        if (this.plotIdx.length > 1) {
            this.plotIdx = new Array<number>(...this.plotIdx)
        }

        this.renderSparkLines()
        this.renderLineChart()
        this.renderSaveButton()
    }

    private onChangeSmoothFocus = () => {
        this.isUpdateDisable = false

        this.focusSmoothed = !this.focusSmoothed;

        this.renderLineChart()
        this.renderSaveButton()
    }

    private onChangeScale = () => {
        this.isUpdateDisable = false

        if (this.currentChart === 1) {
            this.currentChart = 0
        } else {
            this.currentChart = this.currentChart + 1
        }

        this.renderLineChart()
        this.renderSaveButton()
    }
}

export class DistributedMetricsHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('run/:uuid/distributed', [this.handleMetrics])
    }

    handleMetrics = (uuid: string) => {
        SCREEN.setView(new DistributedMetricsView(uuid))
    }
}
