import {CardOptions} from "../../types"
import gpuCache from "./cache"
import {SessionCard} from "../session/card"

export class GPUMemoryCard extends SessionCard {
    constructor(opt: CardOptions) {
        super(opt, 'GPU - Memory', `gpu_memory`, gpuCache, [], false, undefined, 'memory')
    }
}
