const {PI: pi, sqrt, atan} = Math;

const DEFAULT_STROKE_COLOR = '#FFA500'
const DEFAULT_FILL_COLOR = '#000'

export function drawPieCircle(c: CanvasRenderingContext2D, r: number, lineWidth: number, color: string = DEFAULT_STROKE_COLOR) {
    c.fillStyle = color
    c.strokeStyle = color
    c.lineWidth = lineWidth
    c.beginPath()
    c.ellipse(0, 0, r, r, 0, 0, pi * 2)
    c.stroke()

    c.beginPath()
    c.moveTo(0, 0)
    c.lineTo(r, 0)
    c.arc(0, 0, r, 0, pi / 2)
    c.fill()

    c.beginPath()
    c.moveTo(0, 0)
    c.lineTo(0, r)
    c.arc(0, 0, r, -pi / 2, pi, true)
    c.fill()
}

export function drawArmLink(
    c: CanvasRenderingContext2D,
    r1: number,
    r2: number,
    l: number,
    lineWidth: number = 2,
    strokeColor: string = DEFAULT_STROKE_COLOR,
    fillColor: string = DEFAULT_FILL_COLOR) {
    const r = 1.3 * l

    const x = l / 2 + (r1 - r2) * (r1 + r2 + 2 * r) / (2 * l)
    const y = Math.sqrt((r1 + r) ** 2 - x ** 2)

    const a = Math.atan(y / x)
    const b = Math.atan(y / (l - x))

    c.strokeStyle = strokeColor
    c.lineWidth = lineWidth

    c.save()

    c.beginPath()
    c.ellipse(0, 0, r1, r1, 0, a, -a)
    c.arc(x, -y, r, pi - a, b, true)
    c.ellipse(l, 0, r2, r2, 0, b - pi, pi - b)
    c.arc(x, y, r, -b, a - pi, true)
    c.fillStyle = fillColor
    c.fill()
    c.stroke()

    c.restore()
}

function getPiePath(): Path2D {
    const path = new Path2D();

    path.moveTo(-10, 45.8257569)
    path.lineTo(-10, 136.7479433)

    path.arc(-30, 136.747943311, 20, 0, 1.7867568)
    path.arc(0, 0, 160, 1.7867568, 2.9256321)
    path.arc(-136.7479433, 30, 20, 2.9256321, 4.7123889)

    path.lineTo(-45.8257569, 10)

    path.arc(-45.8257569, 20, 10, 4.7123889, 5.8716684)
    path.arc(0, 0, 40, 2.7300758, 1.9823131, true)
    path.arc(-20, 45.8257569, 10, 5.1239058, 0)
    path.closePath()
    return path
}

/**
 *
 * @param c {CanvasRenderingContext2D}
 * @param x {Number}
 * @param y {Number}
 */
export function drawWheel(c: CanvasRenderingContext2D,
                          lineWidth: number = 2,
                          strokeColor = DEFAULT_STROKE_COLOR,
                          fillColor = DEFAULT_FILL_COLOR) {
    const scale = 0.65
    c.save()
    c.scale(scale, scale)
    const path = new Path2D()
    const outerRadius = 180
    path.ellipse(0, 0, outerRadius, outerRadius, 0, 0, 2 * pi, true)
    const piePath = getPiePath()
    path.addPath(piePath)
    path.addPath(piePath, (new DOMMatrix()).rotate(90))
    path.addPath(piePath, (new DOMMatrix()).rotate(180))
    path.addPath(piePath, (new DOMMatrix()).rotate(270))


    c.fillStyle = fillColor
    c.fill(path, 'nonzero')
    c.strokeStyle = strokeColor
    c.lineWidth = lineWidth / scale
    c.stroke(path)
    c.restore()
}

export function drawCirclesWithTangentCone(
    c: CanvasRenderingContext2D,
    r1: number,
    r2: number,
    d: number,
    fillColor: string = DEFAULT_FILL_COLOR) {

    const _sin = (r2 - r1) / d
    const _cos = sqrt(1 - _sin ** 2)
    c.fillStyle = fillColor

    c.beginPath()
    c.ellipse(0, 0, r1, r1, 0, 0, 2 * pi)
    c.fill()

    c.beginPath()
    c.ellipse(d, 0, r2, r2, 0, 0, 2 * pi)
    c.fill()

    c.beginPath()
    c.moveTo(-r1 * _sin, r1 * _cos)
    c.lineTo(-r2 * _sin + d, r2 * _cos)
    c.lineTo(-r2 * _sin + d, -r2 * _cos)

    c.lineTo(-r1 * _sin, -r1 * _cos)
    c.closePath()

    c.fill()
}

