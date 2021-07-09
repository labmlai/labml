import {WeyaElementFunction} from '../../../../lib/weya/weya'
import {Tab} from "./tabs"

export default class KerasCode {
    constructor() {
    }

    render($) {
        $('div.code-sample.bg-dark.px-1.py-2.my-3', $ => {
            $('pre.text-white', $ => {
                $('div', $ => {
                    $('span.key-word', 'from')
                    $('span', ' labml')
                    $('span.key-word', ' import')
                    $('span', ' experiment ')
                    $('span.key-word', 'as ')
                    $('span', 'exp')
                })
                $('div', $ => {
                    $('span.key-word', 'from')
                    $('span', ' labml.utils.keras ')
                    $('span.key-word', 'import')
                    $('span', ' LabMLKerasCallback ')
                })
                $('br')

                $('div', $ => {
                    $('div.labml-api', $ => {
                        $('span.key-word', 'with')
                        $('span', ' exp.record(')
                        $('span.param', 'name=')
                        $('span.string', "'sample'")
                        $('span', ', ')
                        $('span.param', 'exp_conf')
                        $('span', '=conf):')
                    })
                })

                $('div', $ => {
                    new Tab().render($)
                    $('span.key-word', 'for')
                    $('span', ' i ')
                    $('span.key-word', 'in')
                    $('span.built-ins', ' range')
                    $('span', '(')
                    $('span.value', '10000')
                    $('span', ')')
                    $('span', ':')
                })

                $('div', $ => {
                    new Tab().render($)
                    new Tab().render($)
                    $('span', 'model.fit(x_train,')

                })

                $('div', $ => {
                    new Tab().render($)
                    new Tab().render($)
                    new Tab().render($)
                    new Tab().render($)
                    $('span', 'y_train,')
                })
                $('div', $ => {
                    new Tab().render($)
                    new Tab().render($)
                    new Tab().render($)
                    new Tab().render($)
                    $('span.param', 'epochs')
                    $('span', '=conf[')
                    $('span.string', "'epochs'")
                    $('span', ']')
                })
                $('div', $ => {
                    new Tab().render($)
                    new Tab().render($)
                    new Tab().render($)
                    new Tab().render($)
                    $('span.param', 'validation_data')
                    $('span', '=(x_test, y_test),')
                })
                $('div.labml-api', $ => {
                    new Tab().render($)
                    new Tab().render($)
                    new Tab().render($)
                    new Tab().render($)
                    $('span.param', 'callbacks')
                    $('span', '=[LabMLKerasCallback()],')
                })
                $('div', $ => {
                    new Tab().render($)
                    new Tab().render($)
                    new Tab().render($)
                    new Tab().render($)
                    $('span.param', 'verbose')
                    $('span.key-word', '=None')
                    $('span', ')')
                })
            })
        })
    }
}