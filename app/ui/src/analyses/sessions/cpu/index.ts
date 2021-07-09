import {CPUCard} from "./card"
import {CPUHandler} from "./view"
import {Analysis} from "../../types"

let cpuAnalysis: Analysis = {
    card: CPUCard,
    viewHandler: CPUHandler,
    route: 'cpu'
}

export default cpuAnalysis
