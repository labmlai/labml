export const OUTLIER_MARGIN = 0.04
export const BASE_COLOR = '#34495e'


export function getColor(index: number) {
    return CHART_COLORS[index % 10]
}

export function getBaseColor() {
    if (document.body.classList.contains('light')) {
        return '#34495e'
    } else {
        return '#ccc'
    }
}

export const CHART_COLORS = [
    '#4E79A7',
    '#F28E2C',
    '#76B7B2',
    '#E15759',
    '#59A14F',
    '#EDC949',
    '#AF7AA1',
    '#FF9DA7',
    '#9C755F',
    '#BAB0AB'
]

