import {RunConfigsCard} from "./card"
import {RunConfigsHandler} from "./view"
import {Analysis} from "../../types"


let runConfigsAnalysis: Analysis = {
    card: RunConfigsCard,
    viewHandler: RunConfigsHandler,
    route: 'configs'
}

export default runConfigsAnalysis