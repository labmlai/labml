from typing import List, NamedTuple, Dict, Optional

from labml_db import Model, Key, Index

from . import project
from .. import utils


class User(Model['User']):
    name: str
    sub: str
    email: str
    picture: str
    theme: str
    is_dev: bool
    email_verified: bool
    projects: List[Key['project.Project']]

    @classmethod
    def defaults(cls):
        return dict(name='',
                    sub='',
                    email='',
                    picture='',
                    theme='light',
                    is_dev=False,
                    email_verified=False,
                    projects=[]
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


class UserIndex(Index['User']):
    pass


class TokenOwnerIndex(Index['TokenOwner']):
    pass


def get_token_owner(labml_token: str) -> Optional[str]:
    user_key = TokenOwnerIndex.get(labml_token)

    if user_key:
        user = user_key.load()
        return user.email

    return ''


class AuthOInfo(NamedTuple):
    name: str
    sub: str
    email: str
    picture: str
    email_verified: bool


def get_or_create_user(info: AuthOInfo) -> User:
    user_key = UserIndex.get(info.email)

    if not user_key:
        p = project.Project(labml_token=utils.gen_token())
        user = User(name=info.name,
                    sub=info.sub,
                    email=info.email,
                    picture=info.picture,
                    email_verified=info.email_verified,
                    projects=[p.key]
                    )

        user.save()
        p.save()

        UserIndex.set(user.email, user.key)
        project.ProjectIndex.set(p.labml_token, p.key)
        TokenOwnerIndex.set(p.labml_token, user.key)

        return user

    return user_key.load()
