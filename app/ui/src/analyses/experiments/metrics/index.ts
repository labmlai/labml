import {MetricsCard} from "./card"
import {MetricsHandler} from "./view"
import {Analysis} from "../../types"


let metricsAnalysis: Analysis = {
    card: MetricsCard,
    viewHandler: MetricsHandler,
    route: 'metrics'
}

export default metricsAnalysis