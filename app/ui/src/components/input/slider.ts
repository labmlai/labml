import {WeyaElement, WeyaElementFunction} from "../../../../lib/weya/weya"

interface SliderOptions {
    min: number
    max: number
    value: number
    step: number
    onChange: (value: number) => void
}

export class Slider {
    private slider: HTMLInputElement
    private valueElem: HTMLInputElement
    private readonly onChange: (value: number) => void

    private readonly min: number
    private readonly max: number
    private readonly step: number
    value: number

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
            $('div.input-container', $ => {
                $('div.input-content', $ => {
                    this.valueElem = <HTMLInputElement>$('input.value', {
                        type: 'number',
                        value: this.value
                    })
                })
            })
        })

        this.slider.oninput = () => {
            this.value = parseFloat(this.slider.value)
            this.valueElem.value = this.value.toString()
            this.onChange(this.value)
        }
        this.valueElem.oninput = () => {
            let value = parseFloat(this.valueElem.value)
            // check NaN
            if (value !== value) {
                value = this.value
            }

            if (value < this.min) {
                value = this.min
            } else if (value > this.max) {
                value = this.max
            }

            this.value = value

            this.slider.value = this.value.toString()
            this.onChange(this.value)
        }
    }
}