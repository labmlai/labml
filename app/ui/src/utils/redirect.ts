import {NetworkError} from '../network';
import {ROUTER} from '../app';
import {PageNotFoundHandler} from '../views/errors/page_not_found_view'
import {AuthErrorHandler} from '../views/errors/auth_error_view'
import {OtherErrorHandler} from '../views/errors/other_error_view'
import {NetworkErrorHandler} from '../views/errors/network_error_view'
import {DEBUG} from '../env'

export function handleNetworkError(error: Error | NetworkError) {
    if (error instanceof NetworkError) {
        if (error.statusCode === 404 || error.statusCode === 400) {
            ROUTER.navigate('/404')
        } else if (error.statusCode === 401 || error.statusCode === 403) {
            ROUTER.navigate('/401')
        } else {
            OtherErrorHandler.handleOtherError(error)
        }
    } else {
        ROUTER.navigate('/network_error')
    }
}

export function handleNetworkErrorInplace(error: Error | NetworkError) {
    if(DEBUG){
        console.error(error)
    }
    if (error instanceof NetworkError) {
        if (error.statusCode === 404 || error.statusCode === 400) {
            PageNotFoundHandler.handlePageNotFound()
        } else if (error.statusCode === 401 || error.statusCode === 403) {
            AuthErrorHandler.handleAuthError()
        } else {
            OtherErrorHandler.handleOtherError(error)
        }
    } else {
        NetworkErrorHandler.handleNetworkError()
    }
}
