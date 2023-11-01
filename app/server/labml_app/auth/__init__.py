import functools
import json
from typing import Optional

from fastapi import Request

from ..db import project
from ..db import user
from ..db.user import User


def get_app_token(request: Request) -> ('str', User):
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


def get_auth_user(request: Request) -> Optional['user.User']:
    token, u = get_app_token(request)

    return u
