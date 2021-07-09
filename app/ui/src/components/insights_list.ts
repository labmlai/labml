import {WeyaElementFunction} from "../../../lib/weya/weya"
import {InsightModel} from "../models/run"

class Insight {
    className: string
    message: string

    constructor(opt: InsightModel) {
        this.className = '.insight-container'

        if (opt.type === 'danger') {
            this.className += '.danger'
        } else if (opt.type === 'warning') {
            this.className += '.warning'
        } else {
            this.className += '.success'
        }

        this.message = opt.message
    }

    render($: WeyaElementFunction) {
        $('div', this.className, $ => {
            $('span', '.fas.fa-lightbulb.icon', '')
            $('span', '.info', this.message)
        })
    }
}

interface InsightsListOptions {
    insightList: InsightModel[]
}


export default class InsightsList {
    insightList: InsightModel[]

    constructor(opt: InsightsListOptions) {
        this.insightList = opt.insightList
    }

    render($) {
        this.insightList.map((insight, idx) => (
            new Insight({message: insight.message, type: insight.type, time: insight.time}).render($)
        ))
    }
}