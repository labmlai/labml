import {WeyaElementFunction} from '../../../lib/weya/weya'

export default class EmptySessionsList {
    constructor() {
    }

    render($: WeyaElementFunction) {
        $('div.text-center', $ => {
            $('h5.mt-4.px-1', 'You will see your computers here')
            $('p.px-1', 'Run the following command to start monitoring:')
        })

        $('div', $ => {
            $('div.code-sample.bg-dark.px-1.py-2.my-3.code-container', $ => {
                $('pre.text-white', $ => {
                    $('div.labml-api', $ => {
                        $('span.key-word', 'pip')
                        $('span', ' install labml psutil py3nvml')
                        $('br')
                        $('span.key-word', 'labml')
                        $('span', ' monitor')
                    })
                })
            })
        })
    }
}
