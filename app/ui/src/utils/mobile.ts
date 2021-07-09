export function detectMobile(): boolean {
    return /iPhone|iPad|iPod|Android/i.test(navigator.userAgent)
}

let isMobile = detectMobile()

export default isMobile