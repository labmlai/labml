import {DistributedMetricsCard} from "./card"
import {MergedDistributedMetricsHandler} from "./view"
import {Analysis} from "../../types"


let mergedMetricsAnalysis: Analysis = {
    card: DistributedMetricsCard,
    viewHandler: MergedDistributedMetricsHandler,
    route: 'merged_distributed'
}

export default mergedMetricsAnalysis