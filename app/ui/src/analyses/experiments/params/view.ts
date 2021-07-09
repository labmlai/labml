import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {Status} from "../../../models/status"
import CACHE, {AnalysisDataCache, AnalysisPreferenceCache, RunCache, RunStatusCache} from "../../../cache/cache"
import {Run, SeriesModel} from "../../../models/run"
import {AnalysisPreferenceModel} from "../../../models/preferences"
import {DataLoader} from "../../../components/loader"
import {BackButton, SaveButton, ToggleButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import parametersCache from "./cache"
import {LineChart} from "../../../components/charts/lines/chart"
import {getChartType, toPointValues} from "../../../components/charts/utils"
import {SparkLines} from "../../../components/charts/spark_lines/chart"
import {ROUTER, SCREEN} from "../../../app"
import mix_panel from "../../../mix_panel"
import {ViewHandler} from "../../types"
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {setTitle} from '../../../utils/document'
import {ScreenView} from '../../../screen_view'

class ParametersView extends ScreenView {
    elem: HTMLDivElement
    uuid: string
    status: Status
    plotIdx: number[] = []
    currentChart: number
    statusCache: RunStatusCache
    series: SeriesModel[]
    preferenceData: AnalysisPreferenceModel
    analysisCache: AnalysisDataCache
    preferenceCache: AnalysisPreferenceCache
    runHeaderCard: RunHeaderCard
    sparkLines: SparkLines
    lineChartContainer: HTMLDivElement
    sparkLinesContainer: HTMLDivElement
    saveButtonContainer: HTMLDivElement
    toggleButtonContainer: HTMLDivElement
    saveButton: SaveButton
    isUpdateDisable: boolean
    actualWidth: number
    private loader: DataLoader;
    private refresh: AwesomeRefreshButton;
    private runCache: RunCache
    private run: Run

    constructor(uuid: string) {
        super()

        this.uuid = uuid
        this.currentChart = 0
        this.runCache = CACHE.getRun(this.uuid)
        this.statusCache = CACHE.getRunStatus(this.uuid)
        this.analysisCache = parametersCache.getAnalysis(this.uuid)
        this.preferenceCache = parametersCache.getPreferences(this.uuid)

        this.isUpdateDisable = true
        this.saveButton = new SaveButton({onButtonClick: this.updatePreferences, parent: this.constructor.name})

        this.loader = new DataLoader(async (force) => {
            this.status = await this.statusCache.get(force)
            this.run = await this.runCache.get()
            this.series = toPointValues((await this.analysisCache.get(force)).series)
            this.preferenceData = await this.preferenceCache.get(force)
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
        setTitle({section: 'Parameters'})
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
                            width: this.actualWidth
                        })
                        this.runHeaderCard.render($).then()
                        this.toggleButtonContainer = $('div')
                        $('h2', '.header.text-center', 'Parameters - L2 Norm')
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

            setTitle({section: 'Parameters', item: this.run.name})
            this.calcPreferences()

            this.renderSparkLines()
            this.renderLineChart()
            this.renderSaveButton()
            this.renderToggleButton()
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
            this.renderSparkLines()
            this.renderLineChart()
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

    renderSaveButton() {
        this.saveButton.disabled = this.isUpdateDisable
        this.saveButtonContainer.innerHTML = ''
        $(this.saveButtonContainer, $ => {
            this.saveButton.render($)
        })
    }

    renderToggleButton() {
        this.toggleButtonContainer.innerHTML = ''
        $(this.toggleButtonContainer, $ => {
            new ToggleButton({
                onButtonClick: this.onChangeScale,
                text: 'Log',
                isToggled: this.currentChart > 0,
                parent: this.constructor.name
            }).render($)
        })
    }

    renderLineChart() {
        this.lineChartContainer.innerHTML = ''
        $(this.lineChartContainer, $ => {
            new LineChart({
                series: this.series,
                width: this.actualWidth,
                plotIdx: this.plotIdx,
                chartType: getChartType(this.currentChart),
                onCursorMove: [this.sparkLines.changeCursorValues],
                isCursorMoveOpt: true
            }).render($)
        })
    }

    renderSparkLines() {
        this.sparkLinesContainer.innerHTML = ''
        $(this.sparkLinesContainer, $ => {
            this.sparkLines = new SparkLines({
                series: this.series,
                plotIdx: this.plotIdx,
                width: this.actualWidth,
                onSelect: this.toggleChart
            })
            this.sparkLines.render($)
        })
    }

    toggleChart = (idx: number) => {
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

    private calcPreferences() {
        if (this.isUpdateDisable) {
            this.currentChart = this.preferenceData.chart_type

            let analysisPreferences = this.preferenceData.series_preferences
            if (analysisPreferences && analysisPreferences.length > 0) {
                this.plotIdx = [...analysisPreferences]
            } else if (this.series) {
                let res: number[] = []
                for (let i = 0; i < this.series.length; i++) {
                    res.push(i)
                }
                this.plotIdx = res
            }
        }
    }

    onChangeScale = () => {
        this.isUpdateDisable = false

        if (this.currentChart === 1) {
            this.currentChart = 0
        } else {
            this.currentChart = this.currentChart + 1
        }

        this.renderLineChart()
        this.renderSaveButton()
    }

    updatePreferences = () => {
        this.preferenceData.series_preferences = this.plotIdx
        this.preferenceData.chart_type = this.currentChart
        this.preferenceCache.setPreference(this.preferenceData).then()

        this.isUpdateDisable = true
        this.renderSaveButton()
    }
}

export class ParametersHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('run/:uuid/params', [this.handleParameters])
    }

    handleParameters = (uuid: string) => {
        SCREEN.setView(new ParametersView(uuid))
    }
}
