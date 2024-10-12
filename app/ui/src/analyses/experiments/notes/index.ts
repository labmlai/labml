import {NotesCard} from "./card"
import {Analysis} from "../../types"
import {RunHeaderHandler} from "../run_header/view";


let notedAnalysis: Analysis = {
    card: NotesCard,
    viewHandler: RunHeaderHandler,
    route: 'header'
}

export default notedAnalysis