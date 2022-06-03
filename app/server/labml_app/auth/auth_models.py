from pydantic import BaseModel


class SignUpModel(BaseModel):
    email: str
    password: str
    handle: str


class SignInModel(BaseModel):
    email: str
    password: str
