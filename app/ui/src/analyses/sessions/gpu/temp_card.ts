import {CardOptions} from "../../types"
import gpuCache from "./cache"
import {SessionCard} from "../session/card"

export class GPUTempCard extends SessionCard {
    constructor(opt: CardOptions) {
        super(opt, 'GPU - Temperature', `gpu_temp`, gpuCache, [], false, undefined, 'temperature')
    }
}
