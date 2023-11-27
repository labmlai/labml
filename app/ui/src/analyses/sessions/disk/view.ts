import {ROUTER, SCREEN} from "../../../app"
import diskCache from "./cache"
import {ViewHandler} from "../../types"
import {SessionView} from "../session/view"

export class DiskHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('session/:uuid/disk', [this.handleDisk])
    }

    handleDisk = (uuid: string) => {
        SCREEN.setView(new SessionView(uuid, 'Disk', diskCache))
    }
}
