import {Analysis} from "../../types"
import {MetricsSummaryCard} from "./card"


let distributedSummaryAnalysis: Analysis = {
    card: MetricsSummaryCard,
    viewHandler: undefined,
    route: 'distributed'
}

export default distributedSummaryAnalysis