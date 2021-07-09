export async function waitForFrame() {
    return new Promise<void>((resolve) => {
        window.requestAnimationFrame(() => {
            resolve()
        })
    })
}
