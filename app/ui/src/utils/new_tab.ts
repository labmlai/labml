import {UserMessages} from "../components/user_messages"


export function openInNewTab(url, userMessages?: UserMessages) {
    let newWindow = window.open(url, '_blank')

    if (!newWindow || newWindow.closed || typeof newWindow.closed == 'undefined') {
        let error = `cannot open ${url}: blocked pop up windows`
        if (userMessages) {
            userMessages.warning(error)
        } else {
            throw new Error(error)
        }
    } else {
        newWindow.focus()
    }
}