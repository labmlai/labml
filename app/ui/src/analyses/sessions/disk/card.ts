import {CardOptions} from "../../types"
import diskCache from './cache'
import {SessionCard} from "../session/card"

export class DiskCard extends SessionCard {
    constructor(opt: CardOptions) {
        super(opt, 'Disk', 'disk', diskCache, [0, 1], false)
    }
}