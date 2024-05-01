import {Indicator} from "../../../models/run"

export function getSeriesData(series: Indicator[], analysis: string, isMean: boolean = false): Indicator[]{
    let res: Indicator[] = []
    for (let r of series) {
        let s = r.getCopy()
        if (s.name.includes(analysis)) {
            if (s.name.includes('mean')) {
                if (isMean) {
                    s.name = 'mean'
                    return [s]
                } else {
                    continue
                }
            }
            s.name = s.name.replace(/\D/g, '')
            res.push(s)
        }
    }

    return res
}
