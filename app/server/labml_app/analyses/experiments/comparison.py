from typing import Any

from fastapi import Request
from starlette.responses import JSONResponse

from .distributed_metrics import get_merged_metric_tracking_util
from .metrics import MetricsAnalysis, get_metrics_tracking_util, mget
from ..analysis import Analysis
from ...db import run, dist_run


@Analysis.route('POST', 'compare/metrics/{run_uuid}')
async def get_comparison_metrics(request: Request, run_uuid: str) -> Any:
    indicators = (await request.json())['indicators']
    analysis_uuid = dist_run.get_analysis_uuid(run_uuid)

    if analysis_uuid == run_uuid:  # single run
        ans = MetricsAnalysis.get_or_create(run_uuid)
        track_data = ans.get_tracking()
        status_code = 200

        track_data = get_metrics_tracking_util(track_data, indicators)
        response = JSONResponse({'series': track_data})
        response.status_code = status_code

        return response
    else:  # distributed run
        r = dist_run.get(run_uuid)

        metric_list = [MetricsAnalysis(m) if m else None for m in mget(list(r.ranks.values()))]
        metric_list = [m for m in metric_list if m is not None]
        track_data_list = [m.get_tracking() for m in metric_list]

        track_data = get_merged_metric_tracking_util(track_data_list, indicators)

        response = JSONResponse({'series': track_data})
        response.status_code = 200
        return response
