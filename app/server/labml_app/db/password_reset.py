from typing import Optional

import time
from labml_db import Model, Index, Key, load_key

from . import user
from ..utils import gen_token

TIME_TO_EXPIRE = 60 * 60 * 24 * 3


class PasswordReset(Model['PasswordReset']):
    reset_token: str
    user: Key['user.User']
    expire: float

    @classmethod
    def defaults(cls):
        return dict(reset_token='',
                    user=None,
                    expire=None,
                    )

    @property
    def is_valid(self) -> bool:
        return self.expire > time.time()


class PasswordResetIndex(Index['PasswordReset']):
    """rest token"""
    pass


def gen_reset_token(email: str) -> Optional[str]:
    u = user.get_by_email(email)

    if not u:
        return None

    reset_token = gen_token()
    while PasswordResetIndex.get(reset_token) is not None:
        reset_token = gen_token()

    rp = PasswordReset(reset_token=reset_token, user=u.key, expire=time.time() + TIME_TO_EXPIRE)
    rp.save()

    PasswordResetIndex.set(reset_token, rp.key)

    return reset_token


def get(reset_token: str) -> Optional[PasswordReset]:
    return load_key(PasswordResetIndex.get(reset_token))


def delete_token(pr: PasswordReset):
    PasswordResetIndex.delete(pr.reset_token)
    pr.delete()
