import {WeyaElementFunction} from '../../../lib/weya/weya'

export interface SearchOptions {
    onSearch: (query: string) => void
    initText?: string
}

export class SearchView {
    onSearch: () => void
    textbox: HTMLInputElement
    initText: string

    constructor(opt: SearchOptions) {
        this.onSearch = () => {
            opt.onSearch(this.textbox.value)
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
                })
            })
        })

        this.textbox.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                this.onSearch()
            }
        })
    }
}
