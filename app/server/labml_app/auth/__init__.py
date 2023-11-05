import functools
import inspect
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from ..db import project
from ..db import user
from ..db.user import User


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


def _login_not_required():
    u = user.get_or_create_user(identifier='local', is_local_user=True)

    return 'local', u


def get_app_token(request: Request) -> ('str', User):
    token, u = _login_not_required()
    return token, u


def get_auth_user(request: Request) -> Optional['user.User']:
    token, u = get_app_token(request)

    return u
