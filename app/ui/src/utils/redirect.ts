import {NetworkError} from '../network';
import {ROUTER} from '../app';
import {PageNotFoundHandler} from '../views/errors/page_not_found_view'
import {AuthErrorHandler} from '../views/errors/auth_error_view'
import {MiscErrorHandler} from '../views/errors/other_error_view'
import {NetworkErrorHandler} from '../views/errors/network_error_view'
import {DEBUG} from '../env'

export function handleNetworkError(error: Error | NetworkError) {
    if (error instanceof NetworkError) {
        if (error.statusCode === 404 || error.statusCode === 400) {
            ROUTER.navigate('/404')
        } else if (error.statusCode === 401 || error.statusCode === 403) {
            ROUTER.navigate('/401')
        } else {
            MiscErrorHandler.handleMiscError(error)
        }
    } else {
        MiscErrorHandler.handleMiscError(error)
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
            MiscErrorHandler.handleMiscError(error)
        }
    } else {
        NetworkErrorHandler.handleNetworkError(error)
    }
}
