import {WeyaElement, WeyaElementFunction} from "../../../../lib/weya/weya"

interface SliderOptions {
    min: number
    max: number
    value: number
    step: number
    onChange: (value: number) => void
}

export class Slider {
    private valueElem: WeyaElement
    private slider: HTMLInputElement
    private readonly onChange: (value: number) => void

    private readonly min: number
    private readonly max: number
    private readonly step: number
    private value: number

    constructor(opt: SliderOptions) {
        this.onChange = opt.onChange
        this.min = opt.min
        this.max = opt.max
        this.step = opt.step
        this.value = opt.value
    }

    render($: WeyaElementFunction) {
        $('div.slider-container', $ => {
            this.slider = <HTMLInputElement>$('input.slider', {
                type: 'range',
                min: this.min,
                max: this.max,
                value: this.value,
                step: this.step
            })
            this.valueElem = $('span.slider-value', this.value.toFixed(2))
        })

        this.slider.oninput = () => {
            this.value = parseFloat(this.slider.value)
            this.valueElem.textContent = this.slider.value
            this.onChange(this.value)
        }
    }
}