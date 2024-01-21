import asyncio
import inspect
import sys
from typing import Callable, Dict, Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .analyses.experiments import metrics, distributed_metrics
from .analyses.experiments.metrics import MetricsAnalysis
from .logger import logger
from . import settings
from . import auth
from .db import run
from .db import computer
from .db import session
from .db import user
from .db import project
from . import utils
from . import analyses

try:
    import requests
except ImportError:
    pass

EndPointRes = Dict[str, Any]
UNKNOWN_ERROR_MESSAGE = 'Something went wrong. Please try again later. If the problem persists, please reach us via ' \
                        'contact@labml.ai'


def _is_new_run_added(request: Request) -> bool:
    is_run_added = False
    u = auth.get_auth_user(request)
    if u:
        is_run_added = u.default_project.is_run_added

    return is_run_added


async def _update_run(request: Request, labml_token: str, labml_version: str, run_uuid: str, rank: int,
                      world_size: int, main_rank: int):
    errors = []

    token = labml_token

    if len(run_uuid) < 10:
        error = {'error': 'invalid_run_uuid',
                 'message': f'Invalid Run UUID'}
        errors.append(error)
        return {'errors': errors}

    if utils.check_version(labml_version, settings.LABML_VERSION):
        error = {'error': 'labml_outdated',
                 'message': f'Your labml client is outdated, please upgrade: '
                            'pip install labml --upgrade'}
        errors.append(error)
        return {'errors': errors}

    p = project.get_project(labml_token=token)
    if not p:
        token = settings.FLOAT_PROJECT_TOKEN

    r = project.get_run(run_uuid, token)
    if not r and not p:
        if labml_token:
            errors.append({'error': 'invalid_token',
                           'message': 'Please create a valid token at https://app.labml.ai.\n'
                                      'Click on the experiment link to monitor the experiment and '
                                      'add it to your experiments list.'})
        elif not settings.IS_LOCAL_SETUP:
            errors.append({'warning': 'empty_token',
                           'message': 'Please create a valid token at https://app.labml.ai.\n'
                                      'Click on the experiment link to monitor the experiment and '
                                      'add it to your experiments list.'})

    if world_size > 1 and rank > 0:
        run_uuid = f'{run_uuid}_{rank}'

    r = run.get_or_create(request, run_uuid, rank, world_size, main_rank, token)
    s = r.status.load()

    json = await request.json()
    if isinstance(json, list):
        data = json
    else:
        data = [json]

    for d in data:
        r.update_run(d)
        s.update_time_status(d)
        if 'track' in d:
            analyses.AnalysisManager.track(run_uuid, d['track'])

    hp_values = analyses.AnalysisManager.get_experiment_analysis('HyperParamsAnalysis', run_uuid)
    if hp_values is not None:
        hp_values.get_hyper_params()
    else:
        hp_values = {}

    run_uuid = r.url
    if len(run_uuid.split("_")) == 2:
        run_uuid = run_uuid.split("_")[0]

    return {'errors': errors, 'url': f'{request.url.hostname}/{run_uuid}', 'dynamic': hp_values}


async def update_run(request: Request) -> EndPointRes:
    labml_token = request.query_params.get('labml_token', '')
    labml_version = request.query_params.get('labml_version', '')

    run_uuid = request.query_params.get('run_uuid', '')
    rank = int(request.query_params.get('rank', 0))
    world_size = int(request.query_params.get('world_size', 0))
    main_rank = int(request.query_params.get('main_rank', 0))

    res = await _update_run(request, labml_token, labml_version, run_uuid, rank, world_size, main_rank)

    await asyncio.sleep(3)

    return res


async def _update_session(request: Request, labml_token: str, session_uuid: str, computer_uuid: str,
                          labml_version: str):
    errors = []

    token = labml_token

    if len(computer_uuid) < 10:
        error = {'error': 'invalid_computer_uuid',
                 'message': f'Invalid Computer UUID'}
        errors.append(error)
        return {'errors': errors}

    if len(session_uuid) < 10:
        error = {'error': 'invalid_session_uuid',
                 'message': f'Invalid Session UUID'}
        errors.append(error)
        return {'errors': errors}

    if utils.check_version(labml_version, settings.LABML_VERSION):
        error = {'error': 'labml_outdated',
                 'message': f'Your labml client is outdated, please upgrade: '
                            'pip install labml --upgrade'}
        errors.append(error)
        return {'errors': errors}

    p = project.get_project(labml_token=token)
    if not p:
        token = settings.FLOAT_PROJECT_TOKEN

    c = project.get_session(session_uuid, token)
    if not c and not p:
        if labml_token:
            errors.append({'error': 'invalid_token',
                           'message': 'Please create a valid token at https://app.labml.ai.\n'
                                      'Click on the experiment link to monitor the experiment and '
                                      'add it to your experiments list.'})
        elif not settings.IS_LOCAL_SETUP:
            errors.append({'warning': 'empty_token',
                           'message': 'Please create a valid token at https://app.labml.ai.\n'
                                      'Click on the experiment link to monitor the experiment and '
                                      'add it to your experiments list.'})

    c = session.get_or_create(request, session_uuid, computer_uuid, token)
    s = c.status.load()

    json = await request.json()
    if isinstance(json, list):
        data = json
    else:
        data = [json]

    for d in data:
        c.update_session(d)
        s.update_time_status(d)
        if 'track' in d:
            analyses.AnalysisManager.track_computer(session_uuid, d['track'])

    logger.debug(
        f'update_session, session_uuid: {session_uuid}, size : {sys.getsizeof(str(request.json)) / 1024} Kb')

    return {'errors': errors, 'url': f'{request.url.hostname}/{c.url}'}


async def update_session(request: Request) -> EndPointRes:
    labml_token = request.query_params.get('labml_token', '')
    session_uuid = request.query_params.get('session_uuid', '')
    computer_uuid = request.query_params.get('computer_uuid', '')
    labml_version = request.query_params.get('labml_version', '')

    res = await _update_session(request, labml_token, session_uuid, computer_uuid, labml_version)

    await asyncio.sleep(3)

    return res


@auth.login_required
async def claim_run(request: Request, run_uuid: str, token: Optional[str] = None) -> EndPointRes:
    r = run.get(run_uuid)
    u = user.get_by_session_token(token)

    default_project = u.default_project

    if r.run_uuid not in default_project.runs:
        # float_project = project.get_project(labml_token=settings.FLOAT_PROJECT_TOKEN)

        # if r.run_uuid in float_project.runs:
            default_project.runs[r.run_uuid] = r.key
            default_project.is_run_added = True
            default_project.save()
            r.is_claimed = True
            r.owner = u.email
            r.save()

    return {'is_successful': True}


@auth.login_required
async def claim_session(request: Request, session_uuid: str, token: Optional[str] = None) -> EndPointRes:
    c = session.get(session_uuid)
    u = user.get_by_session_token(token)

    default_project = u.default_project

    if c.session_uuid not in default_project.sessions:
        float_project = project.get_project(labml_token=settings.FLOAT_PROJECT_TOKEN)

        if c.session_uuid in float_project.sessions:
            default_project.sessions[c.session_uuid] = c.key
            default_project.save()
            c.is_claimed = True
            c.owner = u.email
            c.save()

    return {'is_successful': True}


async def get_run(request: Request, run_uuid: str) -> JSONResponse:
    run_data = {}
    status_code = 404

    # TODO temporary change to used run_uuid as rank 0
    is_dist_run = len(run_uuid.split("_")) == 1
    run_uuid = utils.get_true_run_uuid(run_uuid)

    r = run.get(run_uuid)
    if r:
        run_data = r.get_data(request, is_dist_run)
        status_code = 200

    response = JSONResponse(run_data)
    response.status_code = status_code

    return response


async def get_session(request: Request, session_uuid: str) -> JSONResponse:
    session_data = {}
    status_code = 404

    c = session.get(session_uuid)
    if c:
        session_data = c.get_data(request)
        status_code = 200

    response = JSONResponse(session_data)
    response.status_code = status_code

    return response


@auth.login_required
async def edit_run(request: Request, run_uuid: str, token: Optional[str] = None) -> EndPointRes:
    r = run.get(run_uuid)
    errors = []

    if r:
        data = await request.json()
        r.edit_run(data)
    else:
        errors.append({'edit_run': 'invalid run uuid'})

    return {'errors': errors}


async def edit_session(request: Request, session_uuid: str) -> EndPointRes:
    c = session.get(session_uuid)
    errors = []

    if c:
        data = await request.json()
        c.edit_session(data)
    else:
        errors.append({'edit_session': 'invalid session_uuid'})

    return {'errors': errors}


async def get_run_status(request: Request, run_uuid: str) -> JSONResponse:
    status_data = {}
    status_code = 404

    if len(run_uuid.split('_')) == 1:  # distributed
        r = run.get(run_uuid)
        if r is not None:
            uuids = [f'{run_uuid}_{i}' for i in range(1, r.world_size)]
            uuids.append(run_uuid)
            status_data = run.get_merged_status_data(uuids)

        if status_data is None or len(status_data.keys()) == 0:
            status_data = {}
            status_code = 404
        else:
            status_code = 200

        response = JSONResponse(status_data)
        response.status_code = status_code

        return response
    else:
        # TODO temporary change to used run_uuid as rank 0
        run_uuid = utils.get_true_run_uuid(run_uuid)

        s = run.get_status(run_uuid)
        if s:
            status_data = s.get_data()
            status_code = 200

        response = JSONResponse(status_data)
        response.status_code = status_code

        return response


async def get_session_status(request: Request, session_uuid: str) -> JSONResponse:
    status_data = {}
    status_code = 404

    s = session.get_status(session_uuid)
    if s:
        status_data = s.get_data()
        status_code = 200

    response = JSONResponse(status_data)
    response.status_code = status_code

    return response


@auth.login_required
@auth.check_labml_token_permission
async def get_runs(request: Request, labml_token: str, token: Optional[str] = None) -> EndPointRes:
    u = user.get_by_session_token(token)

    if labml_token:
        runs_list = run.get_runs(labml_token)
    else:
        default_project = u.default_project
        labml_token = default_project.labml_token
        runs_list = default_project.get_runs()

    run_uuids = []
    dist_run_uuids = []
    for r in runs_list:
        if r.world_size == 0:
            run_uuids.append(r.run_uuid)
        else:
            dist_run_uuids.append(r.run_uuid)

    metric_data = metrics.mget(run_uuids)
    metric_preferences_data = metrics.mget_preferences(run_uuids)

    dist_metric_data = metrics.mget(dist_run_uuids)
    dist_metric_preferences_data = distributed_metrics.mget_preferences(dist_run_uuids)

    metric_data.extend(dist_metric_data)
    metric_preferences_data.extend(dist_metric_preferences_data)

    res = []
    for r, m, mp in zip(runs_list, metric_data, metric_preferences_data):
        s = run.get_status(r.run_uuid)
        if r.run_uuid and r.rank == 0 and m is not None and mp is not None:
            summary = r.get_summary()
            preferences = mp.series_preferences
            metric_values = []

            track_data = MetricsAnalysis(m).get_tracking()

            summary['step'] = 0

            for (idx, series) in enumerate(track_data):
                if len(series['step']) != 0:
                    summary['step'] = max(summary['step'], series['step'][-1])
                if len(preferences) <= idx or preferences[idx] == -1:
                    continue
                metric_values.append({
                    'name': series['name'],
                    'value': series['value'][-1]
                })

            summary['metric_values'] = metric_values

            res.append({**summary, **s.get_data()})

    res = sorted(res, key=lambda i: i['start_time'], reverse=True)

    return {'runs': res, 'labml_token': labml_token}


@auth.login_required
@auth.check_labml_token_permission
async def get_sessions(request: Request, labml_token: str, token: Optional[str] = None) -> EndPointRes:
    u = user.get_by_session_token(token)

    if labml_token:
        sessions_list = session.get_sessions(labml_token)
    else:
        default_project = u.default_project
        labml_token = default_project.labml_token
        sessions_list = default_project.get_sessions()

    res = []
    for c in sessions_list:
        s = session.get_status(c.session_uuid)
        if c.session_uuid:
            res.append({**c.get_summary(), **s.get_data()})

    res = sorted(res, key=lambda i: i['start_time'], reverse=True)

    return {'sessions': res, 'labml_token': labml_token}


@auth.login_required
async def delete_runs(request: Request, token: Optional[str] = None) -> EndPointRes:
    json = await request.json()
    run_uuids = json['run_uuids']

    u = user.get_by_session_token(token)
    u.default_project.delete_runs(run_uuids, u.email)

    return {'is_successful': True}


@auth.login_required
async def delete_sessions(request: Request, token: Optional[str] = None) -> EndPointRes:
    json = await request.json()
    session_uuids = json['session_uuids']

    u = user.get_by_session_token(token)
    u.default_project.delete_sessions(session_uuids, u.email)

    return {'is_successful': True}


@auth.login_required
async def add_run(request: Request, run_uuid: str, token: Optional[str] = None) -> EndPointRes:
    u = user.get_by_session_token(token)

    u.default_project.add_run(run_uuid)

    return {'is_successful': True}


@auth.login_required
async def add_session(request: Request, session_uuid: str, token: Optional[str] = None) -> EndPointRes:
    u = user.get_by_session_token(token)

    u.default_project.add_session(session_uuid)

    return {'is_successful': True}


@auth.login_required
async def get_computer(request: Request, computer_uuid: str) -> EndPointRes:
    c = computer.get_or_create(computer_uuid)

    return c.get_data()


@auth.login_required
async def set_user(request: Request, token: Optional[str] = None) -> EndPointRes:
    u = auth.get_auth_user(request)
    json = await request.json()
    data = json['user']
    if u:
        u.set_user(data)

    return {'is_successful': True}


async def get_user(request: Request):
    token, u = auth.get_app_token(request)
    json = await request.json()
    if json is None:
        return JSONResponse({'is_successful': False, 'error': 'Malformed request'}, status_code=400)
    if u is None:
        return JSONResponse({'is_successful': False, 'error': 'You need to be authenticated to perform this request'},
                            status_code=200)

    session_token = u.generate_session_token(token)
    return JSONResponse({'is_successful': True, 'user': u.get_data(), 'token': session_token}, status_code=200,
                        headers={'Authorization': session_token})


def _add_server(app: FastAPI, method: str, func: Callable, url: str):
    if not inspect.iscoroutinefunction(func):
        raise ValueError(f'{func.__name__} is not a async function')

    app.add_api_route(f'/api/v1/{url}', endpoint=func, methods=[method])


def _add_ui(app: FastAPI, method: str, func: Callable, url: str):
    if not inspect.iscoroutinefunction(func):
        raise ValueError(f'{func.__name__} is not a async function')

    app.add_api_route(f'/api/v1/{url}', endpoint=func, methods=[method])


def add_handlers(app: FastAPI):
    _add_server(app, 'POST', update_run, '{labml_token}/track')
    _add_server(app, 'POST', update_session, '{labml_token}/computer')

    _add_ui(app, 'GET', get_runs, 'runs/{labml_token}')
    _add_ui(app, 'PUT', delete_runs, 'runs')
    _add_ui(app, 'GET', get_sessions, 'sessions/{labml_token}')
    _add_ui(app, 'PUT', delete_sessions, 'sessions')

    _add_ui(app, 'GET', get_computer, 'computer/{computer_uuid}')

    _add_ui(app, 'GET', get_run, 'run/{run_uuid}')
    _add_ui(app, 'POST', edit_run, 'run/{run_uuid}')
    _add_ui(app, 'PUT', add_run, 'run/{run_uuid}/add')
    _add_ui(app, 'PUT', claim_run, 'run/{run_uuid}/claim')
    _add_ui(app, 'GET', get_run_status, 'run/status/{run_uuid}')

    _add_ui(app, 'GET', get_session, 'session/{session_uuid}')
    _add_ui(app, 'POST', edit_session, 'session/{session_uuid}')
    _add_ui(app, 'PUT', add_session, 'session/{session_uuid}/add')
    _add_ui(app, 'PUT', claim_session, 'session/{session_uuid}/claim')
    _add_ui(app, 'GET', get_session_status, 'session/status/{session_uuid}')

    _add_ui(app, 'POST', set_user, 'user')
    _add_ui(app, 'POST', get_user, 'auth/user')

    for method, func, url, login_required in analyses.AnalysisManager.get_handlers():
        if login_required:
            func = func

        _add_ui(app, method, func, url)
