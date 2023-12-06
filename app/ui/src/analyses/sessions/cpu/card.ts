import {CardOptions} from "../../types"
import cpuCache from './cache'
import {SessionCard} from "../session/card"

export class CPUCard extends SessionCard {
    constructor(opt: CardOptions) {
        super(opt, 'CPU', 'cpu', cpuCache, [], true, [0, 100])
    }
}
