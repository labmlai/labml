import {WeyaElementFunction} from '../../../../lib/weya/weya'
import {Tab} from "./tabs"


export default class PyTorchLightningCode {
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
                    $('span', ' labml.utils.lightning ')
                    $('span.key-word', 'import')
                    $('span', ' LabMLLightningLogger ')
                })
                $('br')

                $('div', $ => {
                    $('span', 'trainer = pl.Trainer(')
                    $('span.param', 'gpus')
                    $('span', '=1,')
                })
                $('div', $ => {
                    new Tab().render($)
                    new Tab().render($)
                    new Tab().render($)
                    new Tab().render($)
                    $('span.param', 'max_epochs')
                    $('span', '=5,')
                })
                $('div', $ => {
                    new Tab().render($)
                    new Tab().render($)
                    new Tab().render($)
                    new Tab().render($)
                    $('span.param', 'progress_bar_refresh_rate')
                    $('span', '=20,')
                })
                $('div.labml-api', $ => {
                    new Tab().render($)
                    new Tab().render($)
                    new Tab().render($)
                    new Tab().render($)
                    $('span.param', 'logger')
                    $('span', '=LabMLLighteningLogger())')
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
                        $('span', '=conf, ')
                        $('span.param', 'disable_screen')
                        $('span', '=True):')
                    })
                })
                 $('div.labml-api', $ => {
                    new Tab().render($)
                    new Tab().render($)
                    $('span', 'trainer.fit(model, data_loader)')
                })
            })
        })
    }
}