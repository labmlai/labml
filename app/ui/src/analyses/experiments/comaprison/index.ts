import {ComparisonCard} from "./card"
import {ComparisonHandler} from "./view"
import {Analysis} from "../../types"


let comparisonAnalysis: Analysis = {
    card: ComparisonCard,
    viewHandler: ComparisonHandler,
    route: 'compare'
}

export default comparisonAnalysis
