import {WeyaElementFunction} from '../../../lib/weya/weya'

export interface SearchOptions {
    onSearch: (query: string) => void
    initText?: string
}

export class SearchView {
    onSearch: () => void
    textbox: HTMLInputElement
    inputTimeout: number // NodeJS.Timeout
    initText: string

    constructor(opt: SearchOptions) {
        this.onSearch = () => {
            clearTimeout(this.inputTimeout)
            this.inputTimeout = setTimeout(() => {
                opt.onSearch(this.textbox.value)
            }, 250)
        }
        this.initText = opt.initText ?? ""
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
                    on: {
                        input: this.onSearch
                    }
                })
            })
        })
    }
}
