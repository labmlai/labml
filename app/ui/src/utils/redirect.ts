import {NetworkError} from '../network';
import {ROUTER} from '../app';
import {Sentry} from '../sentry';
import {PageNotFoundHandler} from '../views/errors/page_not_found_view'
import {AuthErrorHandler} from '../views/errors/auth_error_view'
import {OtherErrorHandler} from '../views/errors/other_error_view'
import {NetworkErrorHandler} from '../views/errors/network_error_view'

export function handleNetworkError(error: Error | NetworkError) {
    if (error instanceof NetworkError) {
        if (error.statusCode === 404 || error.statusCode === 400) {
            ROUTER.navigate('/404')
        } else if (error.statusCode === 401 || error.statusCode === 403) {
            ROUTER.navigate('/401')
        } else {
            ROUTER.navigate('/500')
        }
    } else {
        ROUTER.navigate('/network_error')
    }
    Sentry.setExtra('error', error)
    Sentry.captureException(error)
}

export function handleNetworkErrorInplace(error: Error | NetworkError) {
    if (error instanceof NetworkError) {
        if (error.statusCode === 404 || error.statusCode === 400) {
            PageNotFoundHandler.handlePageNotFound()
        } else if (error.statusCode === 401 || error.statusCode === 403) {
            AuthErrorHandler.handleAuthError()
        } else {
            OtherErrorHandler.handleOtherError()
        }
    } else {
        NetworkErrorHandler.handleNetworkError()
    }

    Sentry.setExtra('error', error)
    Sentry.captureException(error)
}
