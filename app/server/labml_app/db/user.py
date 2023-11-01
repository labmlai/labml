import time
from labml_db import Model, Key, Index
from typing import List, Dict, Optional, Set, Any

from . import project
from .. import utils
from ..utils import gen_token

SALT_LENGTH = 16
PROTECTED_KEYS = ['password',
                  'email',
                  'tokens',
                  'session_token_owners',
                  'session_tokens',
                  'is_local_user',
                  'projects',
                  ]
TOKEN_VALIDITY = 30 * 24 * 60 * 60
SESSION_VALIDITY = 60 * 60


class User(Model['User']):
    identifier: str
    name: str
    email: str
    picture: str
    theme: str
    is_dev: bool
    email_verified: bool
    projects: List[Key['project.Project']]
    password: str
    tokens: Dict[str, float]  # Auth Tokens -> Validity & Session Tokens
    session_token_owners: Dict[str, Set[str]]  # auth tokens -> session tokens created with that auth token
    session_tokens: Dict[str, float]
    sub: str
    is_local_user: bool

    @property
    def is_upgraded_user(self):
        if not self.email:
            return False
        if self.password is None:
            return False

        return True

    @classmethod
    def defaults(cls):
        return dict(identifier='',
                    name='',
                    sub='',
                    email='',
                    picture='',
                    password=None,
                    tokens={},
                    session_token_owners={},
                    session_tokens={},
                    theme='light',
                    is_dev=False,
                    email_verified=False,
                    projects=[],
                    is_local_user=False,
                    )

    @property
    def default_project(self) -> 'project.Project':
        return self.projects[0].load()

    def get_data(self) -> Dict[str, any]:
        return {
            'name': self.name,
            'email': self.email,
            'picture': self.picture,
            'theme': self.theme,
            'projects': [p.load().labml_token for p in self.projects],
            'default_project': self.default_project.labml_token
        }

    def set_user(self, data) -> None:
        if 'theme' in data:
            self.theme = data['theme']
            self.save()

    def update_and_save(self, data: Dict[str, Any]):
        res = {k: v for k, v in data.items() if v and k not in PROTECTED_KEYS}

        self.update(res)
        self.save()

    def generate_session_token(self, auth_token: str):
        while True:
            session_token = gen_token()
            result = get_by_session_token(session_token)
            if result is None:
                if self.session_token_owners.get(auth_token) is None:
                    self.session_token_owners[auth_token] = set()
                self.session_token_owners[auth_token].add(session_token)
                self.session_tokens[session_token] = time.time() + SESSION_VALIDITY
                UserSessionTokenIndex.set(session_token, self.key)
                self.save()
                return session_token


class UserIndex(Index['User']):
    pass


class UserEmailIndex(Index['UserEmail']):
    pass


class UserTokenIndex(Index['UserToken']):
    pass


class UserSessionTokenIndex(Index['UserSessionToken']):
    pass


class TokenOwnerIndex(Index['TokenOwner']):
    pass


def get_token_owner(labml_token: str) -> Optional[str]:
    user_key = TokenOwnerIndex.get(labml_token)

    if user_key:
        user = user_key.load()
        return user.email

    return ''


def get_or_create_user(identifier: str, is_local_user=False, **kwargs) -> User:
    user_key = UserIndex.get(identifier)

    if not user_key:
        kwargs.update({'identifier': identifier})
        p = project.Project(labml_token=utils.gen_token())
        u = User(**kwargs)
        u.is_local_user = is_local_user
        u.projects.append(p.key)

        u.save()
        p.save()

        UserIndex.set(u.identifier, u.key)
        project.ProjectIndex.set(p.labml_token, p.key)

    else:
        u = user_key.load()

    if is_local_user:
        u.tokens['local'] = float('inf')
        u.session_token_owners['local'] = u.session_token_owners.get('local', {'local'})
        u.session_tokens['local'] = float('inf')
        UserTokenIndex.set('local', u.key)
        UserSessionTokenIndex.set('local', u.key)
        u.save()

    return u


def get(identifier: str) -> Optional[User]:
    user_key = UserIndex.get(identifier)

    if user_key:
        return user_key.load()

    return None


def get_by_email(email: str) -> Optional[User]:
    user_key = UserEmailIndex.get(email)

    if user_key:
        return user_key.load()

    return None


def get_by_token(token: str) -> Optional[User]:
    user_key = UserTokenIndex.get(token)

    if user_key:
        u: User = user_key.load()
        if u.tokens[token] > time.time():
            return u

        if token in u.session_token_owners:
            for session in u.session_token_owners[token]:
                UserSessionTokenIndex.delete(session)
                if session in u.session_tokens:
                    del u.session_tokens[session]
            del u.session_token_owners[token]
        del u.tokens[token]
        UserTokenIndex.delete(token)
        u.save()
        return None
    return None


def get_by_session_token(token: str) -> Optional[User]:
    user_key = UserSessionTokenIndex.get(token)

    if user_key:
        u: User = user_key.load()
        if u.session_tokens.get(token, 0) > time.time():
            u.session_tokens[token] = time.time() + SESSION_VALIDITY
            u.save()
            return u

        UserSessionTokenIndex.delete(token)
        if token in u.session_tokens:
            del u.session_tokens[token]
        u.save()
        return None

    return None


def get_user_secure(token: str) -> Optional[User]:
    u = get_by_token(token)

    return u


def invalidate_token(token: str) -> bool:
    user = get_by_token(token)
    if user:
        for session in user.session_token_owners[token]:
            UserSessionTokenIndex.delete(session)
            if session in user.session_tokens:
                del user.session_tokens[session]
        del user.session_token_owners[token]
        del user.tokens[token]
        UserTokenIndex.delete(token)
        user.save()
        return True

    return False


def delete(u: User):
    ud = u.get_user_data()

    UserIndex.delete(u.identifier)
    ud.delete()
    u.delete()
