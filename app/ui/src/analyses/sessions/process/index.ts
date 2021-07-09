import {ProcessCard} from "./card"
import {ProcessHandler} from "./view"
import {Analysis} from "../../types"

let processAnalysis: Analysis = {
    card: ProcessCard,
    viewHandler: ProcessHandler,
    route: 'process'
}

export default processAnalysis
