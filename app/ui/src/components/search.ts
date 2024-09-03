import {WeyaElementFunction} from '../../../lib/weya/weya'
import {Loader} from "./loader"

export interface SearchOptions {
    onSearch: (query: string) => void
    initText?: string
}

export class SearchView {
    onSearch: () => void
    textbox: HTMLInputElement
    initText: string
    loader: Loader

    constructor(opt: SearchOptions) {
        this.onSearch = () => {
            opt.onSearch(this.textbox.value)
        }
        this.initText = opt.initText ?? ""
        this.loader = new Loader()
    }

    render($: WeyaElementFunction) {
        $('div', '.search-container.mt-3.mb-3.px-2', $ => {
            $('div', '.search-content', $ => {
                $('span', '.icon', $ => {
                    $('span', '.fas.fa-search', '')
                })
                this.textbox = $('input', '.search-input', {
                    value: this.initText,
                    type: 'search',
                    placeholder: 'Search',
                    'aria-label': 'Search',
                })
                this.loader.render($)
                this.loader.hide(true)
            })
        })

        this.textbox.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                console.log('searching')
                this.loader.hide(false)
                this.onSearch()
                // this.loader.hide(true)
            }
        })
    }
}
