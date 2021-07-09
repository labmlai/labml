import {Run} from '../models/run'

const metaTag = `Monitor PyTorch & TensorFlow model training on mobile phones`

function setMetaDes(des: string) {
    document.getElementsByTagName('meta')
        .namedItem('description')
        .setAttribute('content', des)
}


export function changeRunDec(run: Run) {
    let metaStr = `${metaTag}\n${run.name};${run.comment}\n${run.run_uuid}`

    setMetaDes(metaStr)
}