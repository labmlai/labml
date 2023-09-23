import {Analysis} from "../../types"
import {DistributedMetricsCard} from "./card"
import {MetricsHandler} from "../metrics/view"


let distributedMetricsAnalysis: Analysis = {
    card: DistributedMetricsCard,
    viewHandler: MetricsHandler,
    route: '#'
}

export default distributedMetricsAnalysis