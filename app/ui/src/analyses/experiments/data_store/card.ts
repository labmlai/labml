import {WeyaElementFunction, Weya as $} from '../../../../../lib/weya/weya'
import CACHE, {DataStoreCache} from "../../../cache/cache"
import {Card, CardOptions} from "../../types"
import {DataLoader} from "../../../components/loader"
import {ROUTER} from '../../../app'
import {DataStore} from "../../../models/data_store"
import {DataStoreComponent} from "../../../components/data_store";

export class DataStoreCard extends Card {
    dataStore: DataStore
    dataStoreCache: DataStoreCache
    uuid: string
    width: number
    elem: HTMLDivElement
    private dataStoreContainer: HTMLDivElement
    private loader: DataLoader

    constructor(opt: CardOptions) {
        super(opt)

        this.uuid = opt.uuid
        this.width = opt.width
        this.dataStoreCache = CACHE.getDataStore(this.uuid)
        this.loader = new DataLoader(async (force) => {
            this.dataStore = await this.dataStoreCache.get(force)
        })
    }

    cardName(): string {
        return 'Data Store'
    }

    getLastUpdated(): number {
        return this.dataStoreCache.lastUpdated
    }

    async render($: WeyaElementFunction) {
        this.elem = $('div','.labml-card.labml-card-action.data-store', {on: {click: this.onClick}}, $ => {
            $('h3','.header', 'Data Store')
            this.loader.render($)
            this.dataStoreContainer = $('div')
        })

        try {
            await this.loader.load()

            this.renderDataStore()
        } catch (e) {

        }
    }

    private renderDataStore() {
        $(this.dataStoreContainer, $ => {
          new DataStoreComponent(this.dataStore.filter("")).render($)
        })
    }

    async refresh() {
        try {
            await this.loader.load(true)
            this.renderDataStore()
        } catch (e) {

        }
    }

    onClick = () => {
        ROUTER.navigate(`/run/${this.uuid}/data_store`)
    }
}
