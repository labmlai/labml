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

export interface IsUserLoggedModel {
    is_user_logged: boolean
}

export class IsUserLogged {
    is_user_logged: boolean

    constructor(isUserLogged: IsUserLoggedModel) {
        this.is_user_logged = isUserLogged.is_user_logged
    }
}
