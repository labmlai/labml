import functools
import inspect
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from ..db import app_token
from ..db import project
from ..db import user
from .. import settings


def get_app_token(request: Request) -> 'app_token.AppToken':
    token_id = request.headers.get('Authorization', '')

    return app_token.get_or_create(token_id)


def check_labml_token_permission(func) -> functools.wraps:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        labml_token = kwargs.get('labml_token', '')

        p = project.get_project(labml_token)
        if p and p.is_sharable:
            return func(*args, **kwargs)

        kwargs['labml_token'] = None

        return func(*args, **kwargs)

    return wrapper


def login_required(func) -> functools.wraps:
    @functools.wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        token_id = request.headers.get('Authorization', '')
        at = app_token.get_or_create(token_id)
        if at.is_auth or not settings.IS_LOGIN_REQUIRED:
            if inspect.iscoroutinefunction(func):
                return await func(request, *args, **kwargs)
            else:
                return func(request, *args, **kwargs)
        else:
            response = JSONResponse()
            response.status_code = 403

            return response

    return wrapper


def get_auth_user(request: Request) -> Optional['user.User']:
    s = get_app_token(request)

    u = None
    if s.user:
        u = s.user.load()

    if not settings.IS_LOGIN_REQUIRED:
        u = user.get_or_create_user(
            user.AuthOInfo(**{k: '' for k in ('name', 'email', 'sub', 'email_verified', 'picture')}))

    return u


def get_is_user_logged(request: Request) -> bool:
    s = get_app_token(request)

    if s.is_auth or not settings.IS_LOGIN_REQUIRED:
        return True

    return False
