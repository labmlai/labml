import {CardOptions} from "../../types"
import gpuCache from './cache'
import {SessionCard} from "../session/card"

export class GPUUtilCard extends SessionCard {
    constructor(opt: CardOptions) {
        super(opt, 'GPU - Utilization', `gpu_util`, gpuCache, [], false, undefined, 'utilization')
    }
}


