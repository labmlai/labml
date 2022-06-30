import functools
import inspect
import json
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from .. import settings
from ..db import project
from ..db import user
from ..db.user import User


def get_app_token(request: Request) -> ('str', User):
    if not settings.IS_LOGIN_REQUIRED:
        token, u = _login_not_required()
        return token, u

    res = request.headers.get('Authorization', '')
    try:
        res = json.loads(res)
    except ValueError:
        res = {}

    token = res.get('token', '')
    u = None
    if 'auth' in request.url.path:
        u = user.get_user_secure(token)
    else:
        u = user.get_by_session_token(token)

    return token, u


def check_labml_token_permission(func) -> functools.wraps:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        labml_token = kwargs.get('labml_token', '')

        p = project.get_project(labml_token)
        if p and p.is_sharable:
            return await func(*args, **kwargs)

        kwargs['labml_token'] = None

        return await func(*args, **kwargs)

    return wrapper


# TODO: fix this
def _login_not_required():
    u = user.get_or_create_user(identifier='local', is_local_user=True)

    return 'local', u


def login_required(func) -> functools.wraps:
    @functools.wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        token, u = get_app_token(request)

        kwargs['token'] = token
        if u is None:
            error_end = 'view this content' if request.method == 'GET' else 'perform this action'
            response = JSONResponse({'data': {'error': f'You need to be authorized to {error_end}'}, 'meta': {}})
            response.status_code = 403

            return response

        if inspect.iscoroutinefunction(func):
            res = await func(request, *args, **kwargs)
        else:
            res = func(request, *args, **kwargs)

        if isinstance(res, tuple):
            res, res_code = res
        else:
            res_code = 200 if res else 404

        if res.get('token'):
            token = res['token']

        response = JSONResponse(res)

        response.headers["Authorization"] = f'{token}'
        response.status_code = res_code

        return response

    return wrapper


def get_auth_user(request: Request) -> Optional['user.User']:
    token, u = get_app_token(request)

    return u


def get_is_user_logged(request: Request) -> bool:
    token, u = get_app_token(request)

    if u is None:
        return False

    return True
