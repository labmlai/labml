import {ROUTER, SCREEN} from "../../../app"
import batteryCache from "./cache"
import {ViewHandler} from "../../types"
import {SessionView} from "../session/view"

export class BatteryHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('session/:uuid/battery', [this.handleBattery])
    }

    handleBattery = (uuid: string) => {
        SCREEN.setView(new SessionView(uuid, 'Battery', batteryCache))
    }
}
