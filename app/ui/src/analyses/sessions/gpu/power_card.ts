import {CardOptions} from "../../types"
import gpuCache from "./cache"
import {SessionCard} from "../session/card"

export class GPUPowerCard extends SessionCard {
    constructor(opt: CardOptions) {
        super(opt, 'GPU - Power', `gpu_power`, gpuCache, [], false, undefined, 'power')
    }
}
