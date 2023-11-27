import {ROUTER, SCREEN} from "../../../app"
import gpuCache from "./cache"
import {ViewHandler} from "../../types"
import {SessionView} from "../session/view"


export class GPUTempHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('session/:uuid/gpu_temp', [this.handleGPU])
    }

    handleGPU = (uuid: string) => {
        SCREEN.setView(new SessionView(uuid, 'GPU Temperature', gpuCache, 'temp'))
    }
}
