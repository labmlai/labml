import {SeriesModel} from "../../../models/run"

export interface ProcessModel {
    process_id: string
    name: string
    cpu: SeriesModel
    rss: SeriesModel
    dead: number
    pid: number
}

export interface ProcessDetailsModel extends ProcessModel {
    create_time: number,
    cmdline: string,
    exe: string
    ppid: number
    series: SeriesModel[]
}