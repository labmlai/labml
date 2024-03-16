import {Indicator, IndicatorModel} from "../../../models/run"

export interface ProcessModel {
    process_id: string
    name: string
    cpu: IndicatorModel
    rss: IndicatorModel
    dead: number
    pid: number
}

export interface ProcessDetailsModel {
    create_time: number,
    cmdline: string,
    exe: string
    ppid: number
    series: IndicatorModel[]
    process_id: string
    dead: number
    pid: number
    name: string
}

export interface ProcessDataModel {
    series: ProcessModel[]
    summary: IndicatorModel[]
}

export class ProcessData {
    series: Process[]
    summary: Indicator[]

    constructor(data: ProcessDataModel) {
        this.series = []
        for (let s of data.series) {
            this.series.push(new Process(s))
        }
        this.summary = []
        for (let s of data.summary) {
            this.summary.push(new Indicator(s))
        }
    }
}

export class Process {
    process_id: string
    name: string
    cpu: Indicator
    rss: Indicator
    dead: number
    pid: number

    constructor(process: ProcessModel) {
        this.process_id = process.process_id
        this.name = process.name
        this.cpu = new Indicator(process.cpu)
        this.rss = new Indicator(process.rss)
        this.dead = process.dead
        this.pid = process.pid
    }
}

export class ProcessDetails {
    create_time: number
    cmdline: string
    exe: string
    ppid: number
    series: Indicator[]
    process_id: string
    dead: number
    pid: number
    name: string

    constructor(process: ProcessDetailsModel) {
        this.create_time = process.create_time
        this.cmdline = process.cmdline
        this.exe = process.exe
        this.ppid = process.ppid
        this.dead = process.dead
        this.pid = process.pid
        this.name = process.name

        this.series = []
        for (let s of process.series) {
            this.series.push(new Indicator(s))
        }
    }
}