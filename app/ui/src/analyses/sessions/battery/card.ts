import {Card, CardOptions} from "../../types"
import batteryCache from './cache'
import {SessionCard} from "../session/card"


export class BatteryCard extends SessionCard {
    constructor(opt: CardOptions) {
        super(opt, 'Battery', 'battery', batteryCache, [], true, [0, 100])
    }
}
