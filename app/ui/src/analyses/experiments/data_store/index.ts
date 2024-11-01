import {Analysis} from "../../types"
import {DataStoreCard} from "./card"
import {DataStoreHandler} from "./view"


let dataStoreAnalysis: Analysis = {
    card: DataStoreCard,
    viewHandler: DataStoreHandler,
    route: 'data_store'
}

export default dataStoreAnalysis
