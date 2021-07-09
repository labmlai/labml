import {SessionConfigsCard} from "./card"
import {SessionConfigsHandler} from "./view"
import {Analysis} from "../../types"


let sessionConfigsAnalysis: Analysis = {
    card: SessionConfigsCard,
    viewHandler: SessionConfigsHandler,
    route: 'configs'
}

export default sessionConfigsAnalysis