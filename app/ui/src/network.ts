import {API_BASE_URL, APP_BASE_URL, AUTH0_CLIENT_ID, AUTH0_DOMAIN, MOBILE_APP_NAMESPACE} from './env'
import {User} from './models/user'

class Network {
    baseURL: string

    constructor(baseURL: string) {
        this.baseURL = baseURL
    }

    async getRun(runUUID: string): Promise<any> {
        return this.sendHttpRequest('GET', `/run/${runUUID}`)
    }

    async setRun(runUUID: string, data: object): Promise<any> {
        return this.sendHttpRequest('POST', `/run/${runUUID}`, data)
    }

    async addRun(runUUID: string): Promise<any> {
        return this.sendHttpRequest('PUT', `/run/${runUUID}/add`)
    }

    async claimRun(runUUID: string): Promise<any> {
        return this.sendHttpRequest('PUT', `/run/${runUUID}/claim`)
    }

    async getSession(sessionUUID: string): Promise<any> {
        return this.sendHttpRequest('GET', `/session/${sessionUUID}`)
    }

    async setSession(runUUID: string, data: object): Promise<any> {
        return this.sendHttpRequest('POST', `/session/${runUUID}`, data)
    }

    async addSession(sessionUUID: string): Promise<any> {
        return this.sendHttpRequest('PUT', `/session/${sessionUUID}/add`)
    }

    async claimSession(sessionUUID: string): Promise<any> {
        return this.sendHttpRequest('PUT', `/session/${sessionUUID}/claim`)
    }

    async getRunStatus(runUUID: string): Promise<any> {
        return this.sendHttpRequest('GET', `/run/status/${runUUID}`)
    }

    async getSessionStatus(sessionUUId: string): Promise<any> {
        return this.sendHttpRequest('GET', `/session/status/${sessionUUId}`)
    }

    async getRuns(labml_token: string | null): Promise<any> {
        return this.sendHttpRequest('GET', `/runs/${labml_token}`)
    }

    async getSessions(): Promise<any> {
        return this.sendHttpRequest('GET', `/sessions/${null}`)
    }

    async deleteRuns(runUUIDS: string[]): Promise<any> {
        return this.sendHttpRequest('PUT', `/runs`, {'run_uuids': runUUIDS})
    }

    async deleteSessions(sessionUUIDS: string[]): Promise<any> {
        return this.sendHttpRequest('PUT', `/sessions`, {'session_uuids': sessionUUIDS})
    }

    async getUser(): Promise<any> {
        return this.sendHttpRequest('GET', `/user`, {})
    }

    async setUser(user: User): Promise<any> {
        return this.sendHttpRequest('POST', `/user`, {'user': user})
    }

    async signIn(token: string): Promise<any> {
        let data = {token: token}

        return this.sendHttpRequest('POST', `/auth/sign_in`, data)
    }

    async signOut(): Promise<any> {
        return this.sendHttpRequest('DELETE', `/auth/sign_out`)
    }

    redirectLogin() {
        let redirectURI = `${APP_BASE_URL}/login`
        if (window.localStorage.getItem('platform') === 'cordova') {
            redirectURI = `${MOBILE_APP_NAMESPACE}://${AUTH0_DOMAIN}/cordova/${MOBILE_APP_NAMESPACE}/callback`
        }
        window.location.href = `https://${AUTH0_DOMAIN}/authorize?response_type=token&client_id=${AUTH0_CLIENT_ID}&redirect_uri=${redirectURI}&scope=openid%20profile%20email`
    }

    redirectLogout() {
        window.location.href = `https://${AUTH0_DOMAIN}/v2/logout?client_id=${AUTH0_CLIENT_ID}&returnTo=${APP_BASE_URL}`
    }

    async getIsUserLogged(): Promise<any> {
        return this.sendHttpRequest('GET', `/auth/is_logged`)
    }

    async getAnalysis(url: string, runUUID: string): Promise<any> {
        return this.sendHttpRequest('GET', `/${url}/${runUUID}`, {})
    }

    async getCustomAnalysis(url: string): Promise<any> {
        return this.sendHttpRequest('GET', `/${url}`, {})
    }

    async setAnalysis(url: string, runUUID: string, data): Promise<any> {
        return this.sendHttpRequest('POST', `/${url}/${runUUID}`, data)
    }

    async getPreferences(url: string, runUUID: string): Promise<any> {
        return this.sendHttpRequest('GET', `/${url}/preferences/${runUUID}`, {})
    }

    async updatePreferences(url: string, runUUID: string, data: object): Promise<any> {
        return this.sendHttpRequest('POST', `/${url}/preferences/${runUUID}`, data)
    }

    async startTensorBoard(computerUUId: string, runUUIDs: Array<string>): Promise<any> {
        return this.sendHttpRequest('POST', `/start_tensorboard/${computerUUId}`, {'runs': runUUIDs})
    }

    async clearCheckPoints(computerUUId: string, runUUIDs: Array<string>): Promise<any> {
        return this.sendHttpRequest('POST', `/clear_checkpoints/${computerUUId}`, {'runs': runUUIDs})
    }

    private sendHttpRequest = (method: string, url: string, data: object = {}) => {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest()
            xhr.withCredentials = true
            xhr.open(method, this.baseURL + url)
            xhr.responseType = 'json'

            let authToken = localStorage.getItem('app_token')
            if (authToken) {
                xhr.setRequestHeader('Authorization', authToken)
            }

            if (data) {
                xhr.setRequestHeader('Content-Type', 'application/json')
            }

            xhr.onload = () => {
                if (xhr.status >= 400) {
                    if (xhr.status != 403) {
                        reject(new NetworkError(xhr.status, url, JSON.stringify(xhr.response)))
                    }
                } else {
                    resolve(xhr.response)
                }
            }

            xhr.onerror = () => {
                reject('Network Failure')
            }

            xhr.send(JSON.stringify(data))
        })
    }
}

export class NetworkError {
    statusCode: number
    url: string
    message?: string

    constructor(statusCode: number, url: string, message?: string) {
        this.statusCode = statusCode
        this.url = url
        this.message = message
    }
}

const NETWORK = new Network(API_BASE_URL)
export default NETWORK
