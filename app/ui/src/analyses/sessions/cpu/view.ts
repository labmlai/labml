import {ROUTER, SCREEN} from "../../../app"
import {ViewHandler} from "../../types"
import {SessionView} from "../session/view";
import cpuCache from "./cache";

export class CPUHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('session/:uuid/cpu', [this.handleCPU])
    }

    handleCPU = (uuid: string) => {
        SCREEN.setView(new SessionView(uuid, 'CPU', cpuCache))
    }
}
