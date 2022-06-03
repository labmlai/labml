import {Integrations, Sentry} from './sentry'

import {ROUTER} from './app'
import {RunHandler} from './views/run_view'
import {PageNotFoundHandler} from './views/errors/page_not_found_view'
import {RunsListHandler} from './views/runs_list_view'
import {SessionsListHandler} from './views/sessions_list_view'
import {SignInHandler} from './views/auth/signin_view'
import {SettingsHandler} from './views/settings_view'

import {experimentAnalyses, sessionAnalyses} from "./analyses/analyses"
import {ProcessDetailsHandler} from "./analyses/sessions/process/detail_view"
import {RunHeaderHandler} from "./analyses/experiments/run_header/view"
import {SessionHeaderHandler} from "./analyses/sessions/session_header/view"
import {SessionHandler} from './views/session_view'
import {SENTRY_DSN} from './env'
import {AuthErrorHandler} from './views/errors/auth_error_view'
import {OtherErrorHandler} from './views/errors/other_error_view'
import {NetworkErrorHandler} from './views/errors/network_error_view'
import {SignOutHandler} from './views/auth/signout_view'

ROUTER.route(/^(.*)$/g, [() => {
    ROUTER.navigate('/404')
}])

new SignInHandler()
new SignOutHandler()

new PageNotFoundHandler()
new AuthErrorHandler()
new OtherErrorHandler()
new NetworkErrorHandler()

new RunHandler()
new SessionHandler()
new RunsListHandler()
new SessionsListHandler()
new SettingsHandler()

new RunHeaderHandler()
new SessionHeaderHandler()

//TODO properly import this later
new ProcessDetailsHandler()

ROUTER.route('', [() => {
    ROUTER.navigate('/runs')
}])

ROUTER.route('cordova', [() => {
    window.localStorage.setItem('platform', 'cordova')
    ROUTER.navigate('/runs')
}])

experimentAnalyses.map((analysis, i) => {
    new analysis.viewHandler()
})

sessionAnalyses.map((analysis, i) => {
    new analysis.viewHandler()
})

if (
    document.readyState === 'complete' ||
    document.readyState === 'interactive'
) {
    ROUTER.start(null, false)
} else {
    document.addEventListener('DOMContentLoaded', () => {
        ROUTER.start(null, false)
    })
}

// To make sure that :active is triggered in safari
// Ref: https://developer.apple.com/library/archive/documentation/AppleApplications/Reference/SafariWebContent/AdjustingtheTextSize/AdjustingtheTextSize.html
document.addEventListener("touchstart", () => {
}, true);

if (SENTRY_DSN) {
    Sentry.init({
        dsn: SENTRY_DSN,
        integrations: [
            new Integrations.BrowserTracing(),
        ],
        tracesSampleRate: 1.0,
    })
}

(window as any).handleOpenURL = function (url) {
    window.location.hash = `#${url.split('#')[1]}`
    window.location.reload()
}
