import {CardOptions} from "../../types"
import networkCache from './cache'
import {SessionCard} from "../session/card"

export class NetworkCard extends SessionCard {
    constructor(opt: CardOptions) {
        super(opt, 'Network', 'network', networkCache, [0], false)
    }
}
