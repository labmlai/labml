import asyncio
import inspect
import sys
from typing import Callable, Dict, Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .analyses.experiments import stdout, stderr, stdlogger
from .logger import logger
from . import settings
from . import auth
from .db import run, dist_run
from .db import computer
from .db import session
from .db import user
from .db import project
from .db import status
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

    dr = project.get_dist_run(run_uuid, labml_token=token)
    if not dr and not p:
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

    dr = dist_run.get_or_create(run_uuid, token)
    dr.world_size = world_size
    dr.main_rank = main_rank
    dr.save()

    r = dr.get_or_create_run(rank, request, token)
    s = r.status.load()

    run_uuid = r.run_uuid

    json = await request.json()
    if isinstance(json, list):
        data = json
    else:
        data = [json]

    for d in data:
        r.update_run(d)
        if 'track' in d:
            last_step = analyses.AnalysisManager.track(run_uuid, d['track'])
            s.update_time_status(d, last_step)

        if 'stdout' in d and d['stdout']:
            stdout.update_stdout(run_uuid, d['stdout'])
        if 'stderr' in d and d['stderr']:
            stderr.update_stderr(run_uuid, d['stderr'])
        if 'logger' in d and d['logger']:
            stdlogger.update_std_logger(run_uuid, d['logger'])

    hp_values = analyses.AnalysisManager.get_experiment_analysis('HyperParamsAnalysis', run_uuid)
    if hp_values is not None:
        hp_values.get_hyper_params()
    else:
        hp_values = {}

    app_url = str(request.url).split('api')[0]

    return {'errors': errors, 'url': f'{app_url}run/{dr.uuid}', 'dynamic': hp_values}


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
        s.update_time_status(d, None)
        if 'track' in d:
            analyses.AnalysisManager.track_computer(session_uuid, d['track'])

    logger.debug(
        f'update_session, session_uuid: {session_uuid}, size : {sys.getsizeof(str(request.json)) / 1024} Kb')

    app_url = str(request.url).split('api')[0]

    return {'errors': errors, 'url': f'{app_url}session/{session_uuid}'}


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
    r = dist_run.get(run_uuid)
    u = user.get_by_session_token(token)

    default_project = u.default_project

    if not default_project.is_project_dist_run(run_uuid):
        # float_project = project.get_project(labml_token=settings.FLOAT_PROJECT_TOKEN)

        # if r.run_uuid in float_project.runs:
        default_project.add_dist_run_with_model(r)
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

    dr = dist_run.get(run_uuid)
    if dr is not None:
        main_run_uuid = dr.get_main_uuid()
        r = run.get(main_run_uuid)
    else:
        r = run.get(run_uuid)

    if r:
        run_data = r.get_data(request, parent_uuid=run_uuid)
        run_data['run_uuid'] = run_uuid
        if dr:
            run_data['is_claimed'] = dr.is_claimed
            run_data['owner'] = dr.owner
            run_data['other_rank_run_uuids'] = dr.ranks
        else:
            run_data['is_rank'] = True
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
    json = await request.json()

    u = user.get_by_session_token(token)
    u.default_project.edit_run(run_uuid, json)

    return {'is_successful': True}


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

    dr = dist_run.get(run_uuid)
    if dr is not None:
        status_data = run.get_merged_status_data(list(dr.ranks.values()))
    else:
        r = run.get_status(run_uuid)
        if r:
            status_data = r.get_data()

    if status_data is None or len(status_data.keys()) == 0:
        status_data = {}
        status_code = 404
    else:
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
async def get_runs(request: Request, labml_token: str, token: Optional[str] = None,
                   tag: Optional[str] = None) -> EndPointRes:
    u = user.get_by_session_token(token)

    default_project = u.default_project
    labml_token = default_project.labml_token

    if tag is None:
        runs_list = default_project.get_dist_runs()
    else:
        runs_list = default_project.get_dist_run_by_tags(tag)

    statuses = status.mget([r.status for r in runs_list])
    statuses = {s.key: s for s in statuses}
    res = []
    for r in runs_list:
        s = statuses[r.status]
        res.append({**r.get_summary(), **s.get_data()})

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


async def init_app_api(request: Request):
    client_version = request.query_params.get('version', '')
    api_version = settings.APP_API_VERSION

    if utils.check_version(str(client_version), str(api_version)):
        return JSONResponse({'is_successful': False, 'error': 'API client is outdated, please upgrade'})
    else:
        return JSONResponse({'is_successful': True, 'version': api_version})


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

    _add_ui(app, 'GET', init_app_api, 'init')

    _add_ui(app, 'GET', get_runs, 'runs/{labml_token}/{tag}')
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
