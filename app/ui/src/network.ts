import {API_BASE_URL} from './env'
import {User} from './models/user'

export function getAppToken() {
    return localStorage.getItem('app_token')
}

export function setAppToken(newToken: string) {
    localStorage.setItem('app_token', newToken)
}

class Network {
    baseURL: string
    private sessionToken?: string

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
        let res = await this.sendHttpRequest('POST', `/auth/user`, {
            device: {
                userAgent: navigator.userAgent,
                platform: navigator.platform,
                appName: navigator.appName,
                appCodeName: navigator.appCodeName,
                engine: navigator.product,
                appVersion: navigator.appVersion,
                height: window.screen.height,
                width: window.screen.width
            },
            referrer: window.document.referrer,

        }, false)
        if (res != null && res.user != null && res.user.token != null) {
            setAppToken(res.user.token)
        }
        return res
    }

    async setUser(user: User): Promise<any> {
        return this.sendHttpRequest('POST', `/user`, {'user': user})
    }

    async getAnalysis(url: string, runUUID: string, getAll: boolean = false, currentUUID: string = "",
                      isExperiment: boolean): Promise<any> {
        let method = 'GET'
        if (isExperiment) {
            method = 'POST'
        }

        let data = {
            'get_all': getAll
        }

        return this.sendHttpRequest(method,
            `/${url}/${runUUID}?current=${currentUUID}`, data)
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

    private sendHttpRequest = (method: string, url: string, data: object = {}, retryAuth: boolean = true): Promise<any> => {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest()
            xhr.withCredentials = true
            xhr.open(method, this.baseURL + url)
            xhr.responseType = 'json'

            let appToken = url.includes('/auth/') && !url.includes('/auth/send_verification_email') ? getAppToken() : this.sessionToken
            if (appToken != null) {
                let authDict = {'token': appToken}
                xhr.setRequestHeader('Authorization', JSON.stringify(authDict))
            }

            if (data) {
                xhr.setRequestHeader('Content-Type', 'application/json')
            }

            xhr.onload = async () => {
                let token = xhr.getResponseHeader('Authorization')
                if (token != null) {
                    if (url.includes('/auth/sign_in') || url.includes('/auth/sign_up')) {
                        setAppToken(token)
                    } else {
                        this.updateSession(token)
                    }
                }
                if (xhr.status == 401 && retryAuth) {
                    await this.getUser()
                    try {
                        let res = await this.sendHttpRequest(method, url, data, false)
                        resolve(res)
                    } catch (e) {
                        reject(e)
                    }
                }

                if (xhr.status >= 400) {
                    let errorMessage: string = null
                    if (xhr.response != null) {
                        if (xhr.response.hasOwnProperty('error')) {
                            errorMessage = xhr.response.error
                        } else if (xhr.response.hasOwnProperty('data') && xhr.response.data.hasOwnProperty('error')) {
                            errorMessage = xhr.response.data.error
                        }
                    }
                    if (xhr.status != 403) {
                        reject(new NetworkError(xhr.status, url, JSON.stringify(xhr.response), errorMessage))
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

    private updateSession(token?: string) {
        this.sessionToken = token
    }
}

export class NetworkError {
    statusCode: number
    url: string
    message?: string
    errorDescription?: string
    stackTrace?: string

    constructor(statusCode: number, url: string, message?: string, description?: string) {
        this.statusCode = statusCode
        this.url = url
        this.message = message

        try {
            let jsonMessage = JSON.parse(message)
            this.stackTrace = jsonMessage['trace']
        } catch (e) {
            // there's no stack strace.
        }

        this.errorDescription = description
    }
}

const NETWORK = new Network(API_BASE_URL)
export default NETWORK
