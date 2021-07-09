export function setTitle(opt: { section?: string, item?: string }) {
    if (opt.section != null && opt.item != null) {
        document.title = `${opt.section} - ${opt.item} - labml.ai`
    } else if (opt.section != null || opt.item != null) {
        document.title = `${opt.section || opt.item} - labml.ai`
    } else {
        document.title = 'labml.ai'
    }
}
