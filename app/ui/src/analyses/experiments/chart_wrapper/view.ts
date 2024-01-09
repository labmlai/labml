import {SeriesModel} from "../../../models/run"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {SaveButton, ToggleButton} from "../../../components/buttons"
import {LineChart} from "../../../components/charts/lines/chart"
import {SparkLines} from "../../../components/charts/spark_lines/chart"
import {getChartType} from "../../../components/charts/utils"
import {NumericRangeField} from "../../../components/input/numeric_range_field"

interface ViewWrapperOpt {
    updatePreferences: (data: ViewWrapperData) => void
    onRequestAllMetrics: () => void
    lineChartContainer: HTMLDivElement
    sparkLinesContainer: HTMLDivElement
    saveButtonContainer: WeyaElement
    toggleButtonContainer: WeyaElement
    actualWidth: number

    isUpdateDisable: boolean
}

export interface ViewWrapperData {
    series?: SeriesModel[]
    plotIdx?: number[]
    currentChart?: number
    focusSmoothed?: boolean
    stepRange?: number[]
}

export class ViewWrapper {
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

    private readonly onRequestAllMetrics: () => void

    constructor(opt: ViewWrapperOpt) {
        this.lineChartContainer = opt.lineChartContainer
        this.sparkLinesContainer = opt.sparkLinesContainer
        this.saveButtonContainer = opt.saveButtonContainer
        this.toggleButtonContainer = opt.toggleButtonContainer
        this.actualWidth = opt.actualWidth
        this.onRequestAllMetrics = opt.onRequestAllMetrics

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

    public updateData(data: ViewWrapperData) {
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
                isDistributed: false
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
                isDistributed: false
            })
            this.sparkLines.render($)
        })
    }

    private toggleChart = (idx: number) => {
        this.isUpdateDisable = false

        if (this.plotIdx[idx] > 1) // fix for existing plot idxs
            this.plotIdx[idx] = 1

        if (this.plotIdx[idx] == -1 && this.series[idx].is_summary) {
            // have to load from the backend
            this.onRequestAllMetrics()
        }

        if (this.plotIdx[idx] == 0) {
            this.plotIdx[idx] = 1
        } else if (this.plotIdx[idx] == 1) {
            this.plotIdx[idx] = -1
        } else if (this.plotIdx[idx] == -1) {
            this.plotIdx[idx] = 0
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