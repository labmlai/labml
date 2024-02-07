import {SeriesModel} from "../../../models/run"
import {Weya as $, WeyaElement} from "../../../../../lib/weya/weya"
import {SaveButton, ToggleButton} from "../../../components/buttons"
import {LineChart} from "../../../components/charts/lines/chart"
import {SparkLines} from "../../../components/charts/spark_lines/chart"
import {getChartType} from "../../../components/charts/utils"
import {NumericRangeField} from "../../../components/input/numeric_range_field"
import {Loader} from "../../../components/loader"

interface ViewWrapperOpt {
    dataStore: MetricDataStore
    savePreferences: () => void
    requestMissingMetrics: () => Promise<void>
    lineChartContainer: HTMLDivElement
    sparkLinesContainer: HTMLDivElement
    saveButtonContainer: WeyaElement
    optionRowContainer: WeyaElement
    actualWidth: number
}

export interface MetricDataStore {
    series: SeriesModel[]
    baseSeries?: SeriesModel[]
    plotIdx: number[]
    basePlotIdx?: number[]
    chartType: number
    focusSmoothed: boolean
    stepRange: number[]

    isUnsaved: boolean
}

abstract class ChangeHandlerBase {
    protected wrapper: ViewWrapper

    constructor(wrapper: ViewWrapper) {
        this.wrapper = wrapper
    }

    change() {
        this.handleChange()
        this.wrapper.onChange()
    }

    protected abstract handleChange(): void
}

namespace ChangeHandlers {
    export class StepChangeHandler extends ChangeHandlerBase {
        private readonly stepRangeField: NumericRangeField

        constructor(wrapper: ViewWrapper) {
            super(wrapper)
            this.stepRangeField = wrapper.stepRangeField
        }

        protected handleChange() {
            let [min, max] = this.stepRangeField.getRange()

            this.wrapper.dataStore.stepRange = [min, max]
        }
    }

    export class ScaleChangeHandler extends ChangeHandlerBase {
        protected handleChange() {
            if (this.wrapper.dataStore.chartType === 1) {
                this.wrapper.dataStore.chartType = 0
            } else {
                this.wrapper.dataStore.chartType = this.wrapper.dataStore.chartType + 1
            }
        }
    }

    export class SmoothFocusChangeHandler extends ChangeHandlerBase {
        protected handleChange() {
            this.wrapper.dataStore.focusSmoothed = !this.wrapper.dataStore.focusSmoothed
        }
    }

    export class ToggleChangeHandler extends ChangeHandlerBase {
        private readonly idx: number
        private readonly isBase: boolean

        constructor(wrapper: ViewWrapper, idx: number, isBase: boolean) {
            super(wrapper)
            this.idx = idx
            this.isBase = isBase
        }

        protected handleChange() {
            let plotIdx = this.isBase ? this.wrapper.dataStore.basePlotIdx : this.wrapper.dataStore.plotIdx

            if (plotIdx[this.idx] > 1)
                plotIdx[this.idx] = 1

            if (plotIdx[this.idx] == 0) {
                plotIdx[this.idx] = 1
            } else if (plotIdx[this.idx] == 1) {
                plotIdx[this.idx] = -1
            } else if (plotIdx[this.idx] == -1) {
                plotIdx[this.idx] = 0
            }

            if (this.isBase) {
                this.wrapper.dataStore.basePlotIdx = plotIdx
            } else {
                this.wrapper.dataStore.plotIdx = plotIdx
            }

            if ((this.isBase && this.wrapper.dataStore.baseSeries[this.idx].is_summary) || (!this.isBase && this.wrapper.dataStore.series[this.idx].is_summary)) {
                // have to load from the backend
                this.wrapper.requestMissingMetrics()
            }
        }
    }
}

export class ViewWrapper {
    private readonly lineChartContainer: HTMLDivElement
    private readonly sparkLinesContainer: HTMLDivElement
    private readonly saveButtonContainer: WeyaElement
    private readonly optionRowContainer: WeyaElement

    private readonly actualWidth: number

    public readonly stepRangeField: NumericRangeField
    private readonly saveButton: SaveButton
    private readonly scaleButton: ToggleButton
    private readonly focusButton: ToggleButton
    private sparkLines: SparkLines

    public dataStore: MetricDataStore

    private readonly onRequestMissingMetrics: () => Promise<void>
    private readonly savePreferences: () => void

    constructor(opt: ViewWrapperOpt) {
        this.dataStore = opt.dataStore

        this.lineChartContainer = opt.lineChartContainer
        this.sparkLinesContainer = opt.sparkLinesContainer
        this.saveButtonContainer = opt.saveButtonContainer
        this.optionRowContainer = opt.optionRowContainer
        this.actualWidth = opt.actualWidth
        this.onRequestMissingMetrics = opt.requestMissingMetrics
        this.savePreferences = opt.savePreferences

        this.stepRangeField = new NumericRangeField({
            min: this.dataStore.stepRange[0], max: this.dataStore.stepRange[1],
            onClick: () => {
                let changeHandler = new ChangeHandlers.StepChangeHandler(this)
                changeHandler.change()
            },
            buttonLabel: "Filter Steps"
        })
        this.saveButton = new SaveButton({
            onButtonClick: this.onSave,
            parent: this.constructor.name
        })
        this.scaleButton = new ToggleButton({
            onButtonClick: () => {
                let changeHandler = new ChangeHandlers.ScaleChangeHandler(this)
                changeHandler.change()
            },
            text: 'Log',
            isToggled: this.dataStore.chartType > 0,
            parent: this.constructor.name
        })
        this.focusButton = new ToggleButton({
            onButtonClick: () => {
                let changeHandler = new ChangeHandlers.SmoothFocusChangeHandler(this)
                changeHandler.change()
            },
            text: 'Focus Smoothed',
            isToggled: this.dataStore.focusSmoothed,
            parent: this.constructor.name
        })
    }

    public render(missingBaseExperiment: boolean = false) {
        if (missingBaseExperiment) {
            this.lineChartContainer.innerHTML = ''
            this.sparkLinesContainer.innerHTML = ''
            this.optionRowContainer.innerHTML = ''
            return
        }
        this.renderCharts()
        this.renderSaveButton()
        this.renderOptionRow()
    }

    public onChange() {
        this.dataStore.isUnsaved = true;
        this.renderLineChart();
        this.saveButton.disabled = false;
    }

    public requestMissingMetrics() {
        this.setLoading(true)
        this.onRequestMissingMetrics().then(() => {
            this.renderCharts()
            this.setLoading(false)
        })
    }

    private onSave = () => {
        this.savePreferences()
        this.saveButton.disabled = true
        this.dataStore.isUnsaved = false
    }

    private setLoading(isLoading: boolean) {
        if (isLoading) {
            $(this.lineChartContainer, $ => {
                $('div', '.chart-overlay', $ => {
                    new Loader().render($)
                })
            })
        } else {
            this.renderCharts()
        }
    }

    private renderSaveButton() {
        this.saveButton.disabled = !this.dataStore.isUnsaved
        this.saveButtonContainer.innerHTML = ''
        $(this.saveButtonContainer, $ => {
            this.saveButton.render($)
        })
    }

    private renderCharts() {
        this.renderSparkLines()
        this.renderLineChart()
    }

    private renderOptionRow() {
        this.optionRowContainer.innerHTML = ''
        $(this.optionRowContainer, $ => {
            this.scaleButton.render($)
            this.focusButton.render($)
            this.stepRangeField.render($)
            this.stepRangeField.setRange(this.dataStore.stepRange[0], this.dataStore.stepRange[1])
        })
    }

    private renderLineChart() {
        this.lineChartContainer.innerHTML = ''
        $(this.lineChartContainer, $ => {
            new LineChart({
                series: this.dataStore.series,
                baseSeries: this.dataStore.baseSeries,
                width: this.actualWidth,
                plotIndex: this.dataStore.plotIdx,
                basePlotIdx: this.dataStore.basePlotIdx,
                chartType: getChartType(this.dataStore.chartType),
                onCursorMove: [this.sparkLines.changeCursorValues],
                isCursorMoveOpt: true,
                isDivergent: true,
                stepRange: this.dataStore.stepRange,
                focusSmoothed: this.dataStore.focusSmoothed
            }).render($)
        })
    }

    private renderSparkLines() {
        this.sparkLinesContainer.innerHTML = ''
        $(this.sparkLinesContainer, $ => {
            this.sparkLines = new SparkLines({
                series: this.dataStore.series,
                baseSeries: this.dataStore.baseSeries,
                plotIdx: this.dataStore.plotIdx,
                basePlotIdx: this.dataStore.basePlotIdx,
                width: this.actualWidth,
                onSelect: (idx: number) => {
                    let changeHandler = new ChangeHandlers.ToggleChangeHandler(this, idx, false)
                    changeHandler.change()
                },
                onBaseSelect: (idx: number) => {
                    let changeHandler = new ChangeHandlers.ToggleChangeHandler(this, idx, true)
                    changeHandler.change()
                },
                isDivergent: true
            })
            this.sparkLines.render($)
        })
    }
}