import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {Status} from "../../../models/status"
import CACHE, {AnalysisDataCache, AnalysisPreferenceCache, RunCache, RunStatusCache} from "../../../cache/cache"
import {Run, SeriesModel} from "../../../models/run"
import {BackButton, CustomButton, SaveButton} from "../../../components/buttons"
import {RunHeaderCard} from "../run_header/card"
import hyperParamsCache from "./cache"
import {toPointValues} from "../../../components/charts/utils"
import {EditableSparkLines} from "../../../components/charts/editable_spark_lines/chart"
import {ROUTER, SCREEN} from "../../../app"
import mix_panel from "../../../mix_panel"
import {ViewHandler} from "../../types"
import {DataLoader} from "../../../components/loader"
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {CustomLineChart} from "../../../components/charts/custom_lines/chart"
import {AnalysisPreferenceModel} from "../../../models/preferences"
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {setTitle} from '../../../utils/document'
import {UserMessages} from "../../../components/user_messages"
import {ScreenView} from '../../../screen_view'

class HyperParamsView extends ScreenView {
    elem: HTMLDivElement
    uuid: string
    status: Status
    plotIdx: number[] = []
    series: SeriesModel[]
    run: Run
    runCache: RunCache
    preferenceData: AnalysisPreferenceModel
    analysisCache: AnalysisDataCache
    preferenceCache: AnalysisPreferenceCache
    statusCache: RunStatusCache
    runHeaderCard: RunHeaderCard
    sparkLines: EditableSparkLines
    lineChartContainer: HTMLDivElement
    sparkLinesContainer: HTMLDivElement
    prefsSaveButtonContainer: HTMLDivElement
    paramsSaveButtonContainer: HTMLSpanElement
    paramsResetButtonContainer: HTMLSpanElement
    prefsSaveButton: SaveButton
    paramsSaveButton: CustomButton
    paramsResetButton: CustomButton
    isUpdateDisable: boolean
    actualWidth: number
    private loader: DataLoader
    private refresh: AwesomeRefreshButton
    private isEditMode: boolean
    private userMessages: UserMessages

    constructor(uuid: string) {
        super()

        this.uuid = uuid
        this.statusCache = CACHE.getRunStatus(this.uuid)
        this.analysisCache = hyperParamsCache.getAnalysis(this.uuid)
        this.preferenceCache = hyperParamsCache.getPreferences(this.uuid)
        this.runCache = CACHE.getRun(this.uuid)

        this.prefsSaveButton = new SaveButton({
            onButtonClick: this.updatePreferences.bind(this),
            parent: this.constructor.name
        })
        this.paramsSaveButton = new CustomButton({
            onButtonClick: this.onHyperPramsSave.bind(this),
            parent: this.constructor.name,
            text: 'save'
        })
        this.paramsResetButton = new CustomButton({
            onButtonClick: this.onHyperPramsReset.bind(this),
            parent: this.constructor.name,
            text: 'reset'
        })

        this.userMessages = new UserMessages()

        this.loader = new DataLoader(async (force) => {
            this.status = await this.statusCache.get(force)
            this.run = await this.runCache.get(force)
            this.series = toPointValues((await this.analysisCache.get(force)).series)
            this.preferenceData = await this.preferenceCache.get(force)

            if (this.status && this.status.isRunning) {
                this.isEditMode = true
            }
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
        setTitle({section: 'Dynamic Hyperparameters'})
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', '.page',
                {style: {width: `${this.actualWidth}px`}},
                $ => {
                    this.userMessages.render($)
                    $('div', $ => {
                        $('div', '.nav-container', $ => {
                            new BackButton({text: 'Run', parent: this.constructor.name}).render($)
                            this.prefsSaveButtonContainer = $('div')
                            this.refresh.render($)
                        })
                        this.runHeaderCard = new RunHeaderCard({
                            uuid: this.uuid,
                            width: this.actualWidth
                        })
                        this.runHeaderCard.render($).then()
                        $('h2', '.header.text-center', 'Hyper-Parameters')
                        $('div', '.text-center', $ => {
                            this.paramsResetButtonContainer = $('span', '.show-inline')
                            this.paramsSaveButtonContainer = $('span', '.show-inline')
                        })
                        this.loader.render($)
                        $('div', '.detail-card', $ => {
                            this.sparkLinesContainer = $('div')
                            this.lineChartContainer = $('div')
                        })
                    })
                })
        })

        try {
            await this.loader.load()

            setTitle({section: 'Hyper-Parameters', item: this.run.name})
            this.calcPreferences()

            this.renderSparkLines()
            this.renderLineChart()
            this.renderPrefsSaveButton(true)
            this.renderParamsSaveButton(true)
            this.renderParamsResetButton(true)

            if (!this.run.is_project_run && this.status.isRunning) {
                this.userMessages.warning('Please claim or add this run to change parameters')
            }
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
        } catch (e) {

        } finally {
            if (this.status && !this.status.isRunning) {
                this.refresh.stop()
                this.isEditMode = false
            }
            await this.runHeaderCard.refresh().then()
            this.renderSparkLines()
            this.renderLineChart()
        }
    }

    onVisibilityChange() {
        this.refresh.changeVisibility(!document.hidden)
    }

    renderPrefsSaveButton(isDisabled: boolean = false) {
        this.prefsSaveButton.disabled = isDisabled
        this.prefsSaveButtonContainer.innerHTML = ''
        $(this.prefsSaveButtonContainer, $ => {
            this.prefsSaveButton.render($)
        })
    }

    renderParamsSaveButton(isDisabled: boolean = false) {
        this.paramsSaveButton.disabled = !this.run.is_project_run || isDisabled
        this.paramsSaveButtonContainer.innerHTML = ''
        $(this.paramsSaveButtonContainer, $ => {
            if (this.status.isRunning) {
                this.paramsSaveButton.render($)
            }
        })
    }

    renderParamsResetButton(isDisabled: boolean = false) {
        this.paramsResetButton.disabled = !this.run.is_project_run || isDisabled
        this.paramsResetButtonContainer.innerHTML = ''
        $(this.paramsResetButtonContainer, $ => {
            if (this.status.isRunning) {
                this.paramsResetButton.render($)
            }
        })
    }

    renderLineChart() {
        this.lineChartContainer.innerHTML = ''
        $(this.lineChartContainer, $ => {
            new CustomLineChart({
                series: this.series,
                width: this.actualWidth,
                plotIdx: this.plotIdx,
                onCursorMove: [this.sparkLines.changeCursorValues],
                isCursorMoveOpt: true,
            }).render($)
        })
    }

    renderSparkLines() {
        this.sparkLinesContainer.innerHTML = ''
        $(this.sparkLinesContainer, $ => {
            this.sparkLines = new EditableSparkLines({
                series: this.series,
                plotIdx: this.plotIdx,
                width: this.actualWidth,
                isEditable: this.isEditMode,
                onSelect: this.toggleChart,
                isDivergent: true,
                onEdit: this.onInputChange.bind(this)
            })
            this.sparkLines.render($)
        })
    }

    toggleChart = (idx: number) => {
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
        this.renderPrefsSaveButton()
    }

    async onHyperPramsSave() {
        let res = {}
        let invalids = ''
        for (let sparkLine of this.sparkLines.getEditableSparkLines()) {
            let number = sparkLine.getInput()
            if (number === undefined) {
                continue
            }
            res[sparkLine.name] = number

            let validation = sparkLine.getInputValidation()
            if (validation) {
                invalids += `${sparkLine.name} : ${validation}\n`
            }
        }

        if (invalids) {
            confirm(`${invalids}`)
            this.onHyperPramsReset()
        } else {
            this.renderParamsSaveButton(true)
            this.renderParamsResetButton(true)

            await this.analysisCache.setAnalysis(res)
            this.series = toPointValues((await this.analysisCache.get(true)).series)

            this.renderLineChart()
        }
    }

    onHyperPramsReset() {
        for (let sparkLine of this.sparkLines.getEditableSparkLines()) {
            sparkLine.renderInputValues()
        }

        this.renderParamsSaveButton(true)
        this.renderParamsResetButton(true)
    }

    updatePreferences = () => {
        this.preferenceData.series_preferences = this.plotIdx
        this.preferenceCache.setPreference(this.preferenceData).then()

        this.renderPrefsSaveButton(true)
    }

    onInputChange() {
        this.renderParamsSaveButton(false)
        this.renderParamsResetButton(false)
    }

    private calcPreferences() {
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

export class HyperParamsHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('run/:uuid/hyper_params', [this.handleHyperParams])
    }

    handleHyperParams = (uuid: string) => {
        SCREEN.setView(new HyperParamsView(uuid))
    }
}
