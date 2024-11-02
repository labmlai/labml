import {DataStore} from "../models/data_store"
import {Weya as $, WeyaElement, WeyaElementFunction} from "../../../lib/weya/weya"

export class DataStoreComponent {
    private data: DataStore

    constructor(data: DataStore) {
        this.data = data
    }

    private renderData(element: WeyaElement, data: any, level: number) {
        for (let key in data) {
            let value = data[key]
            if (typeof value === 'object') {
                $(element, $ => {
                    $('div.data-row', $ => {
                        $('span.key' + (level != 0 ? '.sub' : ''), key + ": ")
                        let hr = $('hr')
                        hr.style.opacity = `${0.5 - level * 0.1}`
                    })
                    let elem = $('div')

                    elem.style.marginLeft = `1rem`
                    this.renderData(elem, value, level + 1)
                })
            } else {
                $(element, $ => {
                    $('div.data-row', $ => {
                        $('span.key' +  (level != 0 ? '.sub' : ''), key + ": ")
                        $('span.value', value)
                        let hr = $('hr')
                        hr.style.opacity = `${0.5 - level * 0.1}`
                    })
                })
            }
        }
    }

    render($: WeyaElementFunction) {
        let elem = $(`div.data-container`, $ => {

        })

        this.renderData(elem, this.data.dictionary, 0)
    }
}