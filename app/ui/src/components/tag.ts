import {WeyaElementFunction} from "../../../lib/weya/weya"

interface TagOptions {
    text: string
}

export class TagView {
    text: string

    constructor(opt: TagOptions) {
        this.text = opt.text
    }

    render($: WeyaElementFunction) {
        $('div.tag', this.text)
    }
}