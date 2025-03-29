import {drawPieCircle, drawArmLink, drawWheel} from "./geometry";
import {integrateRK4} from "./ode";

const canvas = document.getElementById('canvas') as HTMLCanvasElement
const ctx = canvas.getContext('2d') as CanvasRenderingContext2D

const {PI: pi, sin} = Math

let [ox, oy] = [0, 0]
let t = Date.now()

interface Params {
    length: number
}

interface State {
    theta: number       // angle of the pendulum
    thetaDot: number    // angular velocity of the pendulum
    phi: number         // angle of the wheel
    phiDot: number      //angular velocity of the wheel
}

let state: State = {
    theta: pi/12,
    thetaDot: 0,
    phi: pi/12,
    phiDot: 0
}

const g = 9.81

const m1 = 0.9
const m2 = 3.0
const r = 0.6
const l = 1.25
const J = m2 * r ** 2
const b = 1.5

const params: Params = {
    length: l * 150
}

function derivate(state: number[], t: number, dt: number) {
    const [_th, _dth, _phi, _dphi] = state
    const friction = -b * (_dphi - _dth)
    const ddth = (-friction - (0.5 * m1 + m2) * l * g * sin(-_th)) / (m1 * l ** 2 / 3 + m2 * l ** 2 + J)
    const ddphi = friction / J
    return [_dth, ddth, _dphi, ddphi]

}

function integrate(state: State, t: number, dt: number): State {
    const next = integrateRK4([state.theta, state.thetaDot, state.phi, state.phiDot], t, dt, derivate)
    return {
        theta: next[0],
        thetaDot: next[1],
        phi: next[2],
        phiDot: next[3]
    }
}

function render() {
    ctx.fillStyle = "#fff"
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    ctx.save()
    ctx.translate(ox, oy)
    ctx.rotate(-pi / 2)
    ctx.rotate(state.theta)
    drawArmLink(ctx, 30, 50, params.length, 2, '#000', '#fff')
    drawPieCircle(ctx, 15, 2, '#000')

    ctx.translate(params.length, 0)
    ctx.rotate(state.phi)
    drawWheel(ctx, 2, '#000', '#fff')
    drawPieCircle(ctx, 15, 2, '#000')
    ctx.restore()

    const now = Date.now()
    const dt = (now - t) / 1000
    state = integrate(state, t, dt)
    t = now

    requestAnimationFrame(render)
}


window.onload = () => {
    [ox, oy] = [canvas.width / 2, canvas.height / 2]
    render()
}

