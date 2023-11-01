import {Analysis} from "../../types"
import {DistributedMetricsCard} from "./card"
import {DistributedMetricsHandler} from "./view"


let distributedMetricsAnalysis: Analysis = {
    card: DistributedMetricsCard,
    viewHandler: DistributedMetricsHandler,
    route: 'distributed'
}

export default distributedMetricsAnalysis