import {SeriesModel} from "../../../models/run"
import CACHE, {AnalysisDataCache, AnalysisPreferenceCache, SessionCache, SessionStatusCache} from "../../../cache/cache"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {Status} from "../../../models/status"
import {DataLoader} from "../../../components/loader"
import {BackButton, SaveButton} from "../../../components/buttons"
import {AnalysisPreferenceModel} from "../../../models/preferences"
import {toPointValues} from "../../../components/charts/utils"
import {SessionHeaderCard} from '../session_header/card'
import {TimeSeriesChart} from '../../../components/charts/timeseries/chart'
import {SparkTimeLines} from '../../../components/charts/spark_time_lines/chart'
import mix_panel from "../../../mix_panel"
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {setTitle} from '../../../utils/document'
import {Session} from '../../../models/session'
import {ScreenView} from '../../../screen_view'
import {AnalysisCache} from "../../helpers"
import {getSeriesData} from "../gpu/utils"
import {DateRangeField} from "../../../components/input/date_range_field"

export class SessionView extends ScreenView {
    elem: HTMLDivElement
    uuid: string
    status: Status
    plotIdx: number[] = []
    statusCache: SessionStatusCache
    series: SeriesModel[]
    preferenceData: AnalysisPreferenceModel
    analysisCache: AnalysisDataCache
    preferenceCache: AnalysisPreferenceCache
    sessionHeaderCard: SessionHeaderCard
    sparkTimeLines: SparkTimeLines
    lineChartContainer: HTMLDivElement
    sparkLinesContainer: HTMLDivElement
    saveButtonContainer: HTMLDivElement
    saveButton: SaveButton
    isUpdateDisable: boolean
    actualWidth: number
    title: string
    subSeries?: string
    optionRow: HTMLDivElement
    private stepRange: number[]
    private stepRangeField: DateRangeField
    private loader: DataLoader
    private refresh: AwesomeRefreshButton
    private sessionCache: SessionCache
    private session: Session

    constructor(uuid: string, title: string, cache: AnalysisCache<any, any>, subSeries?: string) {
        super()

        this.uuid = uuid
        this.sessionCache = CACHE.getSession(this.uuid)
        this.statusCache = CACHE.getSessionStatus(this.uuid)
        this.analysisCache = cache.getAnalysis(this.uuid)
        this.preferenceCache = <AnalysisPreferenceCache>cache.getPreferences(this.uuid)

        this.title = title
        this.subSeries = subSeries

        this.isUpdateDisable = true
        this.saveButton = new SaveButton({onButtonClick: this.updatePreferences, parent: this.constructor.name})

        this.loader = new DataLoader(async (force) => {
            this.status = await this.statusCache.get(force)
            this.session = await this.sessionCache.get()
            if (this.subSeries) {
                this.series = getSeriesData((await this.analysisCache.get(force)).series, this.subSeries)
            } else {
                this.series = toPointValues((await this.analysisCache.get(force)).series)
            }
            this.preferenceData = await this.preferenceCache.get(force)

            let min: number = Number.MAX_VALUE
            let max: number = 0
            for (let s of this.series) {
                min = Math.min(min, s.series[0].step)
                max = Math.max(max, s.series[s.series.length - 1].step)
            }

            this.stepRangeField.setMinMax(new Date(min*1000), new Date(max*1000))
        })
        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))

        this.stepRangeField = new DateRangeField({
            max: 0, min: 0,
            minDate: new Date(0),
            maxDate: new Date(),
            onClick: this.onChangeStepRange.bind(this),
            buttonLabel: "Filter Steps"
        })

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
        setTitle({section: this.title})
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', '.page',
                {style: {width: `${this.actualWidth}px`}},
                $ => {
                    $('div', $ => {
                        $('div', '.nav-container', $ => {
                            new BackButton({text: 'Session', parent: this.constructor.name}).render($)
                            this.saveButtonContainer = $('div')
                            this.refresh.render($)
                        })
                        this.sessionHeaderCard = new SessionHeaderCard({
                            uuid: this.uuid,
                            width: this.actualWidth
                        })
                        this.sessionHeaderCard.render($).then()
                        $('h2', '.header.text-center', this.title)
                        this.loader.render($)
                        $('div', '.detail-card', $ => {
                            this.optionRow = $('div', '.row');
                            this.lineChartContainer = $('div', '.fixed-chart')
                            this.sparkLinesContainer = $('div')
                        })
                    })
                })
        })

        try {
            await this.loader.load()

            setTitle({section: this.title, item: this.session.name})
            this.calcPreferences()

            this.renderSparkLines()
            this.renderLineChart()
            this.renderSaveButton()
            this.renderOptions()

        } catch (e) {
            handleNetworkErrorInplace(e)
        } finally {
            if (this.status && this.status.isRunning) {
                this.refresh.attachHandler(this.sessionHeaderCard.renderLastRecorded.bind(this.sessionHeaderCard))
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
            this.renderOptions()
        } catch (e) {

        } finally {
            if (this.status && !this.status.isRunning) {
                this.refresh.stop()
            }

            await this.sessionHeaderCard.refresh().then()
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

    renderLineChart() {
        this.lineChartContainer.innerHTML = ''
        $(this.lineChartContainer, $ => {
            new TimeSeriesChart({
                series: this.series,
                width: this.actualWidth,
                plotIdx: this.plotIdx,
                onCursorMove: [this.sparkTimeLines.changeCursorValues],
                isCursorMoveOpt: true,
                isDivergent: true,
                stepRange: this.stepRange
            }).render($)
        })
    }

    renderSparkLines() {
        this.sparkLinesContainer.innerHTML = ''
        $(this.sparkLinesContainer, $ => {
            this.sparkTimeLines = new SparkTimeLines({
                series: this.series,
                plotIdx: this.plotIdx,
                width: this.actualWidth,
                onSelect: this.toggleChart,
                isDivergent: true
            })
            this.sparkTimeLines.render($)
        })
    }

    renderOptions() {
        this.optionRow.innerHTML = ''
        this.stepRangeField.setRange(this.stepRange[0], this.stepRange[1])
        $(this.optionRow, $ => {
            this.stepRangeField.render($)
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
        this.renderOptions()
    }


    private onChangeStepRange(min: number, max: number) {
        this.isUpdateDisable = false

        this.stepRange = [min, max]

        this.renderSparkLines()
        this.renderLineChart()
        this.renderSaveButton()
        this.renderOptions()
    }


    calcPreferences() {
        if (this.isUpdateDisable) {
            let analysisPreferences: number[]
            if (this.subSeries) {
                analysisPreferences = this.preferenceData.sub_series_preferences[this.subSeries]
            } else {
                analysisPreferences = this.preferenceData.series_preferences
            }
            if (analysisPreferences && analysisPreferences.length > 0) {
                this.plotIdx = [...analysisPreferences]
            } else if (this.series) {
                let res: number[] = []
                for (let i = 0; i < this.series.length; i++) {
                    res.push(i)
                }
                this.plotIdx = res
            }
            this.stepRange = this.preferenceData.step_range
            this.stepRangeField.setRange(this.stepRange[0], this.stepRange[1])
        }
    }

    updatePreferences = () => {
        if (this.subSeries) {
            this.preferenceData.sub_series_preferences[this.subSeries] = this.plotIdx
        } else {
            this.preferenceData.series_preferences = this.plotIdx
        }
        this.preferenceData.step_range = this.stepRange
        this.preferenceCache.setPreference(this.preferenceData).then()

        this.isUpdateDisable = true
        this.renderSaveButton()
    }
}
