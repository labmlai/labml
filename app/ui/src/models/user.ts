export interface UserModel {
    sub: string
    email: string
    name: string
    picture: string
    theme: string
    email_verified: boolean
    projects: object
    default_project: object
}

export class User {
    sub: string
    email: string
    name: string
    picture: string
    theme: string
    email_verified: boolean
    projects: object
    default_project: object

    constructor(user: UserModel) {
        this.sub = user.sub
        this.email = user.email
        this.name = user.name
        this.picture = user.picture
        this.theme = user.theme
        this.email_verified = user.email_verified
        this.projects = user.projects
        this.default_project = user.default_project
    }
}


export interface SignInModel {
    email: string
    password: string
}

export interface SignUpModel {
    email: string
    password: string
}
