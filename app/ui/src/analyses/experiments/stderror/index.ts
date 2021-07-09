import {StdErrorCard} from "./card"
import {StdErrorHandler} from "./view"
import {Analysis} from "../../types"


let stdErrorAnalysis: Analysis = {
    card: StdErrorCard,
    viewHandler: StdErrorHandler,
    route: 'stderr'
}

export default stdErrorAnalysis