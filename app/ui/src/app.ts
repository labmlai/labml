import {ScreenContainer} from './screen'
import {Router} from '../../lib/weya/router'


export let ROUTER = new Router({
    emulateState: false,
    hashChange: false,
    pushState: true,
    root: '/',
    onerror: e => {
        console.error('Error', e)
    }
})

export let SCREEN = new ScreenContainer()