import {NetworkCard} from "./card"
import {NetworkHandler} from "./view"
import {Analysis} from "../../types"

let networkAnalysis: Analysis = {
    card: NetworkCard,
    viewHandler: NetworkHandler,
    route: 'network'
}

export default networkAnalysis
