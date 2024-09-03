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
        return this.sendHttpRequest('GET', `/run/${runUUID}`)['promise']
    }

    async updateRunData(runUUID: string, data: object): Promise<any> {
        return this.sendHttpRequest('POST', `/run/${runUUID}`, data)['promise']
    }

    async addRun(runUUID: string): Promise<any> {
        return this.sendHttpRequest('PUT', `/run/${runUUID}/add`)['promise']
    }

    async claimRun(runUUID: string): Promise<any> {
        return this.sendHttpRequest('PUT', `/run/${runUUID}/claim`)['promise']
    }

    async getSession(sessionUUID: string): Promise<any> {
        return this.sendHttpRequest('GET', `/session/${sessionUUID}`)['promise']
    }

    async setSession(runUUID: string, data: object): Promise<any> {
        return this.sendHttpRequest('POST', `/session/${runUUID}`, data)['promise']
    }

    async addSession(sessionUUID: string): Promise<any> {
        return this.sendHttpRequest('PUT', `/session/${sessionUUID}/add`)['promise']
    }

    async claimSession(sessionUUID: string): Promise<any> {
        return this.sendHttpRequest('PUT', `/session/${sessionUUID}/claim`)['promise']
    }

    async getRunStatus(runUUID: string): Promise<any> {
        return this.sendHttpRequest('GET', `/run/status/${runUUID}`)['promise']
    }

    async getSessionStatus(sessionUUId: string): Promise<any> {
        return this.sendHttpRequest('GET', `/session/status/${sessionUUId}`)['promise']
    }

    async getRuns(tag: string): Promise<any> {
        return this.sendHttpRequest('GET', `/runs/${null}${tag ? `/${tag}` : ""}`)['promise']
    }

    async getSessions(): Promise<any> {
        return this.sendHttpRequest('GET', `/sessions/${null}`)['promise']
    }

    async archiveRuns(runUUIDS: string[]): Promise<any> {
        return this.sendHttpRequest('POST', `/runs/archive`, {'run_uuids': runUUIDS})['promise']
    }

    async unarchiveRuns(runUUIDS: string[]): Promise<any> {
        return this.sendHttpRequest('POST', `/runs/unarchive`, {'run_uuids': runUUIDS})['promise']
    }

    async deleteRuns(runUUIDS: string[]): Promise<any> {
        return this.sendHttpRequest('PUT', `/runs`, {'run_uuids': runUUIDS})['promise']
    }

    async deleteSessions(sessionUUIDS: string[]): Promise<any> {
        return this.sendHttpRequest('PUT', `/sessions`, {'session_uuids': sessionUUIDS})['promise']
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

        }, false)['promise']
        if (res != null && res.user != null && res.user.token != null) {
            setAppToken(res.user.token)
        }
        return res
    }

    async setUser(user: User): Promise<any> {
        return this.sendHttpRequest('POST', `/user`, {'user': user})['promise']
    }

    getAnalysis(url: string, runUUID: string, data: object,): {promise: Promise<any>, xhr: XMLHttpRequest} {
        let method = 'GET'
        if (data != null) {
            method = 'POST'
        }

        return this.sendHttpRequest(method,
            `/${url}/${runUUID}`, data)
    }

    async getCustomAnalysis(url: string): Promise<any> {
        return this.sendHttpRequest('GET', `/${url}`, {})['promise']
    }

    async setAnalysis(url: string, runUUID: string, data): Promise<any> {
        return this.sendHttpRequest('POST', `/${url}/${runUUID}`, data)['promise']
    }

    async getPreferences(url: string, runUUID: string): Promise<any> {
        return this.sendHttpRequest('GET', `/${url}/preferences/${runUUID}`, {})['promise']
    }

    async updatePreferences(url: string, runUUID: string, data: object): Promise<any> {
        return this.sendHttpRequest('POST', `/${url}/preferences/${runUUID}`, data)['promise']
    }

    async createMagicMetric(runUUID: string): Promise<any> {
        return this.sendHttpRequest('GET', `/custom_metrics/${runUUID}/magic`, {})['promise']
    }

    async createCustomMetric(runUUID: string, data: object): Promise<any> {
        return this.sendHttpRequest('POST', `/custom_metrics/${runUUID}/create`, data)['promise']
    }

    async getCustomMetrics(runUUID: string): Promise<any> {
        return this.sendHttpRequest('GET', `/custom_metrics/${runUUID}`, {})['promise']
    }

    async deleteCustomMetric(runUUID: string, metricUUID: string): Promise<any> {
        return this.sendHttpRequest('POST', `/custom_metrics/${runUUID}/delete`, {id: metricUUID})['promise']
    }

    async updateCustomMetric(runUUID: string, data: object): Promise<any> {
        return this.sendHttpRequest('POST', `/custom_metrics/${runUUID}`, data)['promise']
    }

    async getLogs(runUUID: string, url: string, pageNo: any): Promise<any> {
        return this.sendHttpRequest('POST', `/logs/${url}/${runUUID}`, {
            'page': pageNo
        })['promise']
    }

    async updateLogOptions(runUUID: string, url: string, wrapLogs: boolean): Promise<any> {
        return this.sendHttpRequest('POST', `/logs/${url}/${runUUID}/opt`, {
            'wrap_logs': wrapLogs
        })['promise']
    }

    private sendHttpRequest = (method: string, url: string, data: object = {}, retryAuth: boolean = true): {promise: Promise<any>, xhr: XMLHttpRequest} => {
        const xhr = new XMLHttpRequest()
        let promise = new Promise((resolve, reject) => {
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
                    reject(new NetworkError(xhr.status, url, JSON.stringify(xhr.response), errorMessage))
                } else {
                    resolve(xhr.response)
                }
            }

            xhr.onerror = (event) => {
                reject(new NetworkError(xhr.status, url, xhr.responseText,
                    `XHR request failed: ${event}\n Type: ${event.type}: ${event.loaded} bytes transferred`))
            }

            xhr.send(JSON.stringify(data))
        })

        return {'promise': promise, 'xhr': xhr}
    }

    private updateSession(token?: string) {
        this.sessionToken = token
    }
}

export interface ErrorResponse {
    is_successful: boolean
    error?: string
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

    toString() {
        return `Status Code: ${this.statusCode}\n
        URL: ${this.url}\n
        Description: ${this.errorDescription}\n
        Message: ${this.message}\n
        StackTrace: ${this.stackTrace}`
    }
}

const NETWORK = new Network(API_BASE_URL)
export default NETWORK
