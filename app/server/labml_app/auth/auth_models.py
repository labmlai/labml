from pydantic import BaseModel


class SignUpModel(BaseModel):
    name: str
    email: str
    password: str


class SignInModel(BaseModel):
    email: str
    password: str
