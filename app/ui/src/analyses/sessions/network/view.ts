import {ROUTER, SCREEN} from "../../../app"
import networkCache from "./cache"
import {ViewHandler} from "../../types"
import {SessionView} from "../session/view"

export class NetworkHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('session/:uuid/network', [this.handleNetwork])
    }

    handleNetwork = (uuid: string) => {
        SCREEN.setView(new SessionView(uuid, 'Network', networkCache))
    }
}
