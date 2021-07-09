import {BatteryCard} from "./card"
import {BatteryHandler} from "./view"
import {Analysis} from "../../types"

let batteryAnalysis: Analysis = {
    card: BatteryCard,
    viewHandler: BatteryHandler,
    route: 'battery'
}

export default batteryAnalysis
