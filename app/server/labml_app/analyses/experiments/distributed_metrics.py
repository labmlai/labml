from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from ..analysis import Analysis
from . import metrics
from ...db import run


@Analysis.route('GET', 'distributed/metrics/{run_uuid}')
async def get_dist_metrics_tracking(request: Request, run_uuid: str) -> Any:
    track_data = []
    status_code = 404

    r = run.get(run_uuid)

    run_uuids = []
    for i in range(r.world_size):
        if i == 0:
            run_uuids.append(run_uuid)
        else:
            run_uuids.append(f'{run_uuid}_{i}')

    metric_analyses = metrics.mget(run_uuids)
    for i, ma in enumerate(metric_analyses):
        if ma:
            track_data.append(metrics.MetricsAnalysis(ma).get_tracking())
            status_code = 200
        else:
            track_data.append([])

    response = JSONResponse({'series': track_data, 'insights': []})
    response.status_code = status_code

    return response
