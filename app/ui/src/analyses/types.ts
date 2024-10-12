import {WeyaElementFunction} from "../../../lib/weya/weya"

export interface CardOptions {
    uuid: string
    width: number
    params?: Record<string, string>
}

export abstract class Card {
    protected constructor(opt: CardOptions) {
    }

    abstract render($: WeyaElementFunction)

    abstract refresh()

    abstract getLastUpdated(): number

    abstract cardName(): string
}

export abstract class ViewHandler {

}

export interface Analysis {
    card: new (opt: CardOptions) => Card
    viewHandler: new () => ViewHandler
    route: string
}



