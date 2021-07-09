import {WeyaElementFunction} from '../../../../lib/weya/weya'
import {Tab} from "./tabs"

export default class PyTorchCode {
    constructor() {
    }

    render($: WeyaElementFunction) {
        $('div.code-sample.bg-dark.px-1.py-2.my-3', $ => {
            $('pre.text-white', $ => {
                $('div', $ => {
                    $('span.key-word', 'from')
                    $('span', ' numpy.random ')
                    $('span.key-word', 'import')
                    $('span', ' random')
                })
                $('div', $ => {
                    $('span.key-word', 'from')
                    $('span', ' labml')
                    $('span.key-word', ' import')
                    $('span', ' tracker, experiment ')
                    $('span.key-word', 'as ')
                    $('span', 'exp')
                })
                $('br')

                $('div', $ => {
                    $('span', 'conf = {')
                    $('span.string', "'batch_size")
                    $('span', ':')
                    $('span.value', '20')
                    $('span', '}')
                })
                $('br')

                $('div', $ => {
                    $('span.key-word', 'def ')
                    $('span.method', 'train')
                    $('span', '(n: ')
                    $('span.built-ins', 'int')
                    $('span', '):')
                })
                $('div', $ => {
                    new Tab().render($)
                    $('span', 'loss = ')
                    $('span.value', '0.999')
                    $('span', ' ** n + random() / ')
                    $('span.value', '10')
                })

                $('div', $ => {
                    new Tab().render($)
                    $('span', 'accuracy = ')
                    $('span.value', '1')
                    $('span', ' - ')
                    $('span.value', '0.999')
                    $('span', ' ** n + random() / ')
                    $('span.value', '10')
                })
                $('br')

                $('div', $ => {
                    new Tab().render($)
                    $('span.key-word', 'return')
                    $('span', ' loss, accuracy')
                })
                $('br')

                $('div.labml-api', $ => {
                    $('span.key-word', 'with')
                    $('span', ' exp.record(')
                    $('span.param', 'name=')
                    $('span.string', "'sample'")
                    $('span', ' ,')
                    $('span.param', 'exp_conf')
                    $('span', '=conf):')
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
                    $('span', 'lss, acc = train(i)')
                })

                $('div.labml-api', $ => {
                    new Tab().render($)
                    new Tab().render($)

                    $('span', 'tracker.save(i, ')
                    $('span.param', 'loss')
                    $('span', '=lss, ')
                    $('span.param', 'accuracy')
                    $('span', '=acc)')
                })
            })
        })
    }
}