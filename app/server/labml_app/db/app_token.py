import time
from uuid import uuid4

from labml_db import Model, Key, Index

from . import user

EXPIRATION_DELAY = 60 * 60 * 24 * 30


def gen_token_id() -> str:
    return uuid4().hex


def gen_expiration() -> float:
    return time.time() + EXPIRATION_DELAY


class AppToken(Model['Session']):
    token_id: str
    expiration: float
    user: Key['user.User']
    is_local_user: bool

    @classmethod
    def defaults(cls):
        return dict(token_id='',
                    expiration='',
                    user=None,
                    is_local_user=False,
                    )

    @property
    def is_auth(self) -> bool:
        if self.user is None:
            return False
        if self.is_local_user:
            return True

        return self.expiration > time.time()


class AppTokenIndex(Index['AppToken']):
    pass


def get_or_create(token_id: str, is_local_user: bool = False) -> AppToken:
    if not token_id:
        token_id = gen_token_id()

    app_token_key = AppTokenIndex.get(token_id)

    if not app_token_key:
        app_token = AppToken(token_id=token_id,
                             expiration=gen_expiration(),
                             is_local_user=is_local_user,
                             )
        app_token.save()
        AppTokenIndex.set(app_token.token_id, app_token.key)

        return app_token

    return app_token_key.load()


def delete(app_token: AppToken) -> None:
    AppTokenIndex.delete(app_token.token_id)
    app_token.delete()
