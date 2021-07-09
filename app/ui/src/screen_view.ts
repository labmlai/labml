import {WeyaElement} from '../../lib/weya/weya'

abstract class ScreenView {
    get requiresAuth() {
        return true
    }

    abstract render(): WeyaElement

    onResize(width: number) {
    }

    destroy() {
    }

    onRefresh() {
    }

    onVisibilityChange() {
    }
}

export {ScreenView}
