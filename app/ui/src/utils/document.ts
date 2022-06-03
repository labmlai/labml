export function setTitle(opt: { section?: string, item?: string }) {
    if (opt.section != null && opt.item != null) {
        document.title = `${opt.section} - ${opt.item} - labml.ai`
    } else if (opt.section != null || opt.item != null) {
        document.title = `${opt.section || opt.item} - labml.ai`
    } else {
        document.title = 'labml.ai'
    }
}

export function getPath() {
    return window.location.pathname + window.location.search
}

export function clearChildElements(elem: HTMLElement) {
    // Comparison: https://www.measurethat.net/Benchmarks/Show/13770/0/innerhtml-vs-innertext-vs-removechild-vs-remove#latest_results_block
    while (elem.firstChild) {
        elem.firstChild.remove()
    }
}
