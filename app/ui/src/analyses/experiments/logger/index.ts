import {LoggerCard} from "./card"
import {LoggerHandler} from "./view"
import {Analysis} from "../../types"


let loggerAnalysis: Analysis = {
    card: LoggerCard,
    viewHandler: LoggerHandler,
    route: 'logger'
}

export default loggerAnalysis