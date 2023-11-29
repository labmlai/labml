import {Card, CardOptions} from "../../types"
import memoryCache from './cache'
import {SessionCard} from "../session/card"

export class MemoryCard extends SessionCard {
    constructor(opt: CardOptions) {
        super(opt, 'Memory', 'memory', memoryCache, [0,1], false)
    }
}
