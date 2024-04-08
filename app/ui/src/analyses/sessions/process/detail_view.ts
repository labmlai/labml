import {ProcessDetails} from "./types"
import CACHE, {SessionStatusCache} from "../../../cache/cache"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {Status} from "../../../models/status"
import {DataLoader} from "../../../components/loader"
import {ROUTER, SCREEN} from "../../../app"
import {BackButton} from "../../../components/buttons"
import {processDetailsCache} from "./cache"
import {SessionHeaderCard} from '../session_header/card'
import {AwesomeRefreshButton} from '../../../components/refresh_button'
import {handleNetworkErrorInplace} from '../../../utils/redirect'
import {setTitle} from '../../../utils/document'
import {DetailsDataCache} from "./cache_helper"
import EditableField from "../../../components/input/editable_field"
import {formatTime} from "../../../utils/time"
import {SingleScaleLineChart} from "../../../components/charts/timeseries/single_scale_lines"
import {SparkTimeLines} from "../../../components/charts/spark_time_lines/chart"
import {ScreenView} from '../../../screen_view'
import {Indicator} from "../../../models/run"

class ProcessDetailView extends ScreenView {
    elem: HTMLDivElement
    uuid: string
    actualWidth: number
    processId: string
    status: Status
    statusCache: SessionStatusCache
    processData: ProcessDetails
    series: Indicator[]
    plotIdx: number[] = []
    analysisCache: DetailsDataCache
    sessionHeaderCard: SessionHeaderCard
    sparkTimeLines: SparkTimeLines
    private fieldContainer: HTMLDivElement
    private lineChartContainer: HTMLDivElement
    private sparkLinesContainer: HTMLDivElement
    private loader: DataLoader
    private refresh: AwesomeRefreshButton

    constructor(uuid: string, processId: string) { // session uuid, process id
        super()

        this.uuid = uuid
        this.processId = processId
        this.statusCache = CACHE.getSessionStatus(this.uuid)
        this.analysisCache = processDetailsCache.getAnalysis(this.uuid, this.processId)

        this.loader = new DataLoader(async (force) => {
            this.status = await this.statusCache.get(force)
            this.processData = (await this.analysisCache.get(force))

            this.series = this.processData.series
        })
        this.refresh = new AwesomeRefreshButton(this.onRefresh.bind(this))

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
        setTitle({section: 'Processes Details'})
        this.elem.innerHTML = ''
        $(this.elem, $ => {
            $('div', '.page',
                {style: {width: `${this.actualWidth}px`}},
                $ => {
                    $('div', $ => {
                        $('div', '.nav-container', $ => {
                            new BackButton({text: 'Session', parent: this.constructor.name}).render($)
                            this.refresh.render($)
                        })
                        this.sessionHeaderCard = new SessionHeaderCard({
                            uuid: this.uuid,
                            width: this.actualWidth
                        })
                        this.sessionHeaderCard.render($).then()
                        $('h2', '.header.text-center', 'Processes Details')
                        this.loader.render($)
                        this.fieldContainer = $('div', '.input-list-container')
                        $('div', '.detail-card', $ => {
                            this.lineChartContainer = $('div', '.fixed-chart')
                            this.sparkLinesContainer = $('div')
                        })
                    })
                })
        })

        try {
            await this.loader.load()

            setTitle({section: 'Processes Details', item: this.processData.name})
            this.calcPreferences()

            this.renderFields()
            this.renderSparkLines()
            this.renderLineChart()
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

    renderFields() {
        this.fieldContainer.innerHTML = ''
        $(this.fieldContainer, $ => {
            $('ul', $ => {
                new EditableField({
                    name: 'Name',
                    value: this.processData.name,
                }).render($)
                new EditableField({
                    name: 'Created Time',
                    value: formatTime(this.processData.create_time),
                }).render($)
                new EditableField({
                    name: 'PID',
                    value: this.processData.pid.toString(),
                }).render($)
                new EditableField({
                    name: 'CMDLINE',
                    value: this.processData.cmdline,
                }).render($)
                new EditableField({
                    name: 'EXE',
                    value: this.processData.exe,
                }).render($)
            })
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
    }

    renderLineChart() {
        this.lineChartContainer.innerHTML = ''
        $(this.lineChartContainer, $ => {
            new SingleScaleLineChart({
                series: this.series,
                width: this.actualWidth,
                plotIdx: this.plotIdx,
                onCursorMove: [this.sparkTimeLines.changeCursorValues],
                isCursorMoveOpt: true,
                isDivergent: true
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

    calcPreferences() {
        if (this.series) {
            let res: number[] = []
            for (let i = 0; i < this.series.length; i++) {
                res.push(i)
            }
            this.plotIdx = res
        }
    }
}

export class ProcessDetailsHandler {
    constructor() {
        ROUTER.route('session/:uuid/process/:processId', [this.handleProcessDetails])
    }

    handleProcessDetails = (uuid: string, processId: string) => {
        SCREEN.setView(new ProcessDetailView(uuid, processId))
    }
}
