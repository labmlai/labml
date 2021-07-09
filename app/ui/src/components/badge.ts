import {WeyaElementFunction} from "../../../lib/weya/weya"

interface BadgeViewOptions {
    text: string
}

export class BadgeView {
    text: string

    constructor(opt: BadgeViewOptions) {
        this.text = opt.text
    }

    render($: WeyaElementFunction) {
        $('div.label.label-info', this.text)
    }
}