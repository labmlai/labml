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
        this.sub = user?.sub
        this.email = user?.email
        this.name = user?.name
        this.picture = user?.picture
        this.theme = user?.theme
        this.email_verified = user?.email_verified
        this.projects = user?.projects
        this.default_project = user?.default_project
    }

    get is_complete(): boolean {
        if (this.email == null) {
            return false
        }
        if (this.name == null) {
            return false
        }
        return this.default_project != null
    }
}

export interface SignInModel {
    email: string
    password: string
}

export interface SignUpModel {
    name: string
    email: string
    password: string
}

export interface PasswordResetModel {
    reset_token: string
    new_password: string
}
