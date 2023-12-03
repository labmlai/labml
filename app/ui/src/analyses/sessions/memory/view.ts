import {ROUTER, SCREEN} from "../../../app"
import memoryCache from "./cache"
import {ViewHandler} from "../../types"
import {SessionView} from "../session/view"

export class MemoryHandler extends ViewHandler {
    constructor() {
        super()
        ROUTER.route('session/:uuid/memory', [this.handleMemory])
    }

    handleMemory = (uuid: string) => {
        SCREEN.setView(new SessionView(uuid, 'Memory', memoryCache))
    }
}
