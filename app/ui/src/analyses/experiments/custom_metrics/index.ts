import {MetricCard} from "./card"
import {MetricHandler} from "./view"
import {Analysis} from "../../types"


let metricAnalysis: Analysis = {
    card: MetricCard,
    viewHandler: MetricHandler,
    route: 'metric'
}

export default metricAnalysis
