import {ParametersCard} from "./card"
import {ParametersHandler} from "./view"
import {Analysis} from "../../types"


let parametersAnalysis: Analysis = {
    card: ParametersCard,
    viewHandler: ParametersHandler,
    route: 'params'
}

export default parametersAnalysis