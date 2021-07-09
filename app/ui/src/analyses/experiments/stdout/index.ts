import {StdOutCard} from "./card"
import {StdOutHandler} from "./view"
import {Analysis} from "../../types"


let stdOutAnalysis: Analysis = {
    card: StdOutCard,
    viewHandler: StdOutHandler,
    route: 'stdout'
}

export default stdOutAnalysis