import asyncio
import inspect
import sys
from typing import Callable, Dict, Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .logger import logger
from . import settings
from . import auth
from .db import run, password_reset
from .db import computer
from .db import session
from .db import user
from .db import project
from .db import blocked_uuids
from .db import job
from . import utils
from . import analyses
from .auth.auth_models import SignInModel, SignUpModel

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


@utils.analytics.AnalyticsEvent.time_this(None)
async def sign_in(request: Request, data: SignInModel):
    authenticated_token = user.authenticate(data.email, data.password)
    if authenticated_token is None:
        return JSONResponse({'is_successful': False, 'error': 'The email or password is incorrect'}, status_code=401)

    u = user.get_user_secure(authenticated_token)
    utils.analytics.AnalyticsEvent.people_set(identifier=u.email, first_name=u.name, last_name='', email=u.email)

    return JSONResponse({'is_successful': True, 'token': authenticated_token}, status_code=200,
                        headers={'Authorization': authenticated_token})


@utils.analytics.AnalyticsEvent.time_this(None)
async def sign_up(request: Request, data: SignUpModel):
    u = user.get_or_create_user(utils.gen_token())
    if not u:
        return JSONResponse({'is_successful': False, 'error': 'Bots are not allowed to sign up'}, status_code=401)
    result = u.upgrade_user(data.name, data.email, data.password)
    if result == -1:
        return JSONResponse({'is_successful': False, 'error': 'The account is already created'}, status_code=405)
    elif result == -2:
        return JSONResponse({'is_successful': False, 'error': 'The email is already in use'}, status_code=422)
    elif result == -4:
        return JSONResponse({'is_successful': False, 'error': 'The handle is already in use'}, status_code=422)
    elif result != 0:
        return JSONResponse({'is_successful': False, 'error': UNKNOWN_ERROR_MESSAGE}, status_code=500)

    authenticated_token = user.authenticate(data.email, data.password)
    utils.analytics.AnalyticsEvent.people_set(identifier=u.email, first_name=u.name, last_name='', email=u.email)

    return JSONResponse({'is_successful': True, 'token': authenticated_token}, status_code=200,
                        headers={'Authorization': authenticated_token})


@utils.analytics.AnalyticsEvent.time_this(None)
async def reset_password(request: Request):
    json = await request.json()
    if 'reset_token' not in json or 'new_password' not in json:
        return JSONResponse({'is_successful': False, 'error': 'Malformed request'}, status_code=400)

    reset_token = json['reset_token']
    new_password = json['new_password']

    rp = password_reset.get(reset_token)
    if not rp:
        return JSONResponse({'is_successful': False,
                             'error': 'The reset token is either invalid or expired. Please request a new password '
                                      'reset link.'},
                            status_code=401)
    if not rp.is_valid:
        password_reset.delete_token(rp)
        return JSONResponse({'is_successful': False,
                             'error': 'The reset token is either invalid or expired. Please request a new password '
                                      'reset link.'},
                            status_code=401)

    # should not happen
    if rp.reset_token != reset_token:
        return JSONResponse({'is_successful': False,
                             'error': 'Internal Error. If the problem persist, please reach us out at contact@labml.ai'},
                            status_code=500)

    u = rp.user.load()
    u.reset_password(new_password)
    password_reset.delete_token(rp)

    return JSONResponse({'is_successful': True})


@auth.login_required
@utils.analytics.AnalyticsEvent.time_this(None)
async def sign_out(request: Request, token: Optional[str] = None) -> EndPointRes:
    return {'is_successful': user.invalidate_token(token), 'token': ''}


@utils.analytics.AnalyticsEvent.time_this(0.4)
async def _update_run(request: Request, labml_token: str, run_uuid: str, labml_version: str):
    errors = []

    token = labml_token

    if blocked_uuids.is_run_blocked(run_uuid):
        error = {'error': 'blocked_run_uuid',
                 'message': f'Blocked or deleted run, uuid:{run_uuid}'}
        errors.append(error)
        return {'errors': errors}

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

    r = run.get_or_create(request, run_uuid, token)
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

    if r.is_sync_needed or not r.is_in_progress:
        c = computer.get_or_create(r.computer_uuid)
        try:
            c.create_job(job.JobMethods.CALL_SYNC, {})
        except AssertionError as e:
            logger.debug(f'error while creating CALL_SYNC : {e}')

    hp_values = analyses.AnalysisManager.get_experiment_analysis('HyperParamsAnalysis', run_uuid).get_hyper_params()

    return {'errors': errors, 'url': r.url, 'dynamic': hp_values}


async def update_run(request: Request) -> EndPointRes:
    labml_token = request.query_params.get('labml_token', '')
    run_uuid = request.query_params.get('run_uuid', '')
    labml_version = request.query_params.get('labml_version', '')

    res = await _update_run(request, labml_token, run_uuid, labml_version)

    await asyncio.sleep(3)

    return res


@utils.analytics.AnalyticsEvent.time_this(0.4)
async def _update_session(request: Request, labml_token: str, session_uuid: str, computer_uuid: str,
                          labml_version: str):
    errors = []

    token = labml_token

    if blocked_uuids.is_session_blocked(session_uuid):
        error = {'error': 'blocked_session_uuid',
                 'message': f'Blocked or deleted session, uuid:{session_uuid}'}
        errors.append(error)
        return {'errors': errors}

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

    return {'errors': errors, 'url': c.url}


async def update_session(request: Request) -> EndPointRes:
    labml_token = request.query_params.get('labml_token', '')
    session_uuid = request.query_params.get('session_uuid', '')
    computer_uuid = request.query_params.get('computer_uuid', '')
    labml_version = request.query_params.get('labml_version', '')

    res = await _update_session(request, labml_token, session_uuid, computer_uuid, labml_version)

    await asyncio.sleep(3)

    return res


@auth.login_required
@utils.analytics.AnalyticsEvent.time_this(None)
async def claim_run(request: Request, run_uuid: str, token: Optional[str] = None) -> EndPointRes:
    r = run.get(run_uuid)
    u = user.get_by_session_token(token)

    default_project = u.default_project

    if r.run_uuid not in default_project.runs:
        float_project = project.get_project(labml_token=settings.FLOAT_PROJECT_TOKEN)

        if r.run_uuid in float_project.runs:
            default_project.runs[r.run_uuid] = r.key
            default_project.is_run_added = True
            default_project.save()
            r.is_claimed = True
            r.owner = u.email
            r.save()

            utils.analytics.AnalyticsEvent.track(request, 'run_claimed', {'run_uuid': r.run_uuid})
            utils.analytics.AnalyticsEvent.run_claimed_set(u.email)

    return {'is_successful': True}


@auth.login_required
@utils.analytics.AnalyticsEvent.time_this(None)
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

            utils.analytics.AnalyticsEvent.track(request, 'session_claimed', {'session_uuid': c.session_uuid})
            utils.analytics.AnalyticsEvent.computer_claimed_set(u.email)

    return {'is_successful': True}


@utils.analytics.AnalyticsEvent.time_this(None)
async def get_run(request: Request, run_uuid: str) -> JSONResponse:
    run_data = {}
    status_code = 404

    r = run.get(run_uuid)
    if r:
        run_data = r.get_data(request)
        status_code = 200

    response = JSONResponse(run_data)
    response.status_code = status_code

    return response


@utils.analytics.AnalyticsEvent.time_this(None)
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


@utils.analytics.AnalyticsEvent.time_this(None)
async def get_run_status(request: Request, run_uuid: str) -> JSONResponse:
    status_data = {}
    status_code = 404

    s = run.get_status(run_uuid)
    if s:
        status_data = s.get_data()
        status_code = 200

    response = JSONResponse(status_data)
    response.status_code = status_code

    return response


@utils.analytics.AnalyticsEvent.time_this(None)
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
@utils.analytics.AnalyticsEvent.time_this(None)
@auth.check_labml_token_permission
async def get_runs(request: Request, labml_token: str, token: Optional[str] = None) -> EndPointRes:
    print(request.url.hostname)
    u = user.get_by_session_token(token)

    if labml_token:
        runs_list = run.get_runs(labml_token)
    else:
        default_project = u.default_project
        labml_token = default_project.labml_token
        runs_list = default_project.get_runs()

    res = []
    for r in runs_list:
        s = run.get_status(r.run_uuid)
        if r.run_uuid:
            res.append({**r.get_summary(), **s.get_data()})

    res = sorted(res, key=lambda i: i['start_time'], reverse=True)

    return {'runs': res, 'labml_token': labml_token}


@auth.login_required
@utils.analytics.AnalyticsEvent.time_this(None)
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


@utils.analytics.AnalyticsEvent.time_this(None)
@auth.login_required
async def delete_runs(request: Request, token: Optional[str] = None) -> EndPointRes:
    json = await request.json()
    run_uuids = json['run_uuids']

    u = user.get_by_session_token(token)
    u.default_project.delete_runs(run_uuids, u.email)

    return {'is_successful': True}


@utils.analytics.AnalyticsEvent.time_this(None)
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
@utils.analytics.AnalyticsEvent.time_this(None)
async def get_computer(request: Request, computer_uuid: str) -> EndPointRes:
    c = computer.get_or_create(computer_uuid)

    return c.get_data()


@auth.login_required
@utils.analytics.AnalyticsEvent.time_this(None)
async def set_user(request: Request, token: Optional[str] = None) -> EndPointRes:
    u = auth.get_auth_user(request)
    json = await request.json()
    data = json['user']
    if u:
        u.set_user(data)

    return {'is_successful': True}


@utils.analytics.AnalyticsEvent.time_this(None)
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


@utils.analytics.AnalyticsEvent.time_this(None)
async def is_user_logged(request: Request) -> EndPointRes:
    return {'is_user_logged': auth.get_is_user_logged(request)}


@utils.analytics.AnalyticsEvent.time_this(None)
async def sync_computer(request: Request) -> EndPointRes:
    """End point to sync UI-server and UI-computer. runs: to sync with the server.
        """
    errors = []

    computer_uuid = request.query_params.get('computer_uuid', '')
    if len(computer_uuid) < 10:
        error = {'error': 'invalid_computer_uuid',
                 'message': f'Invalid Computer UUID'}
        errors.append(error)
        return {'errors': errors}

    c = computer.get_or_create(computer_uuid)

    json = await request.json()
    runs = json.get('runs', [])
    res = c.sync_runs(runs)

    return {'runs': res}


@utils.analytics.AnalyticsEvent.time_this(60.4)
async def polling(request: Request) -> EndPointRes:
    """End point to sync UI-server and UI-computer. jobs: statuses of jobs.
    pending jobs will be returned in the response if there any
           """
    errors = []

    computer_uuid = request.query_params.get('computer_uuid', '')
    if len(computer_uuid) < 10:
        error = {'error': 'invalid_computer_uuid',
                 'message': f'Invalid Computer UUID'}
        errors.append(error)
        return {'errors': errors}

    c = computer.get_or_create(computer_uuid)

    c.update_last_online()

    json = await request.json()
    job_responses = json.get('jobs', [])
    if job_responses:
        c.sync_jobs(job_responses)

    pending_jobs = []
    for i in range(16):
        c = computer.get_or_create(computer_uuid)
        pending_jobs = c.get_pending_jobs()
        if pending_jobs:
            break

        await asyncio.sleep(3)

    return {'jobs': pending_jobs}


@auth.login_required
@utils.analytics.AnalyticsEvent.time_this(30.4)
async def start_tensor_board(request: Request, computer_uuid: str, token: Optional[str] = None) -> EndPointRes:
    """End point to start TB for set of runs. runs: all the runs should be from a same computer.
            """
    c = computer.get_or_create(computer_uuid)

    if not c.is_online:
        return {'status': job.JobStatuses.COMPUTER_OFFLINE, 'data': {}}

    json = await request.json()
    runs = json.get('runs', [])
    j = c.create_job(job.JobMethods.START_TENSORBOARD, {'runs': runs})

    for i in range(15):
        c = computer.get_or_create(computer_uuid)
        completed_job = c.get_completed_job(j.job_uuid)
        if completed_job and completed_job.is_completed:
            return completed_job.to_data()

        await asyncio.sleep(2)

    data = j.to_data()
    data['status'] = job.JobStatuses.TIMEOUT

    return data


@auth.login_required
@utils.analytics.AnalyticsEvent.time_this(30.4)
async def clear_checkpoints(request: Request, computer_uuid: str, token: Optional[str] = None) -> EndPointRes:
    """End point to clear checkpoints for set of runs. runs: all the runs should be from a same computer.
            """
    c = computer.get_or_create(computer_uuid)

    if not c.is_online:
        return {'status': job.JobStatuses.COMPUTER_OFFLINE, 'data': {}}

    json = await request.json()
    runs = json.get('runs', [])
    j = c.create_job(job.JobMethods.CLEAR_CHECKPOINTS, {'runs': runs})

    for i in range(15):
        c = computer.get_or_create(computer_uuid)
        completed_job = c.get_completed_job(j.job_uuid)
        if completed_job and completed_job.is_completed:
            return completed_job.to_data()

        await asyncio.sleep(2)

    data = j.to_data()
    data['status'] = job.JobStatuses.TIMEOUT

    return data


def _add_server(app: FastAPI, method: str, func: Callable, url: str):
    if not inspect.iscoroutinefunction(func):
        raise ValueError(f'{func.__name__} is not a async function')

    app.add_api_route(f'/api/v1/{url}', endpoint=func, methods=[method])


def _add_ui(app: FastAPI, method: str, func: Callable, url: str):
    if not inspect.iscoroutinefunction(func):
        raise ValueError(f'{func.__name__} is not a async function')

    app.add_api_route(f'/api/v1/{url}', endpoint=func, methods=[method])


def add_handlers(app: FastAPI):
    _add_server(app, 'POST', update_run, 'track')
    _add_server(app, 'POST', update_session, 'computer')
    _add_server(app, 'POST', sync_computer, 'sync')
    _add_server(app, 'POST', polling, 'polling')

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
    _add_ui(app, 'POST', sign_in, 'auth/sign_in')
    _add_ui(app, 'POST', sign_up, 'auth/sign_up')
    _add_ui(app, 'POST', sign_out, 'auth/sign_out')
    _add_ui(app, 'POST', reset_password, 'auth/password_reset')
    _add_ui(app, 'GET', is_user_logged, 'auth/is_logged')

    _add_ui(app, 'POST', start_tensor_board, 'start_tensorboard/{computer_uuid}')
    _add_ui(app, 'POST', clear_checkpoints, 'clear_checkpoints/{computer_uuid}')

    for method, func, url, login_required in analyses.AnalysisManager.get_handlers():
        if login_required:
            func = auth.login_required(func)

        _add_ui(app, method, func, url)
