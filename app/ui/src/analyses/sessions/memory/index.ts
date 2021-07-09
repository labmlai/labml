import {MemoryCard} from "./card"
import {MemoryHandler} from "./view"
import {Analysis} from "../../types"

let memoryAnalysis: Analysis = {
    card: MemoryCard,
    viewHandler: MemoryHandler,
    route: 'memory'
}

export default memoryAnalysis
