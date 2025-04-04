import * as ort from 'onnxruntime-web';

import {
    drawPieCircle,
    drawArmLink,
    drawWheel,
    drawCirclesWithTangentCone
} from "./geometry";
import {integrateRK4Async} from "./ode";
import {Chart, ChartConfiguration} from "chart.js/auto";

const canvas = document.getElementById('canvas') as HTMLCanvasElement
const ctx = canvas.getContext('2d') as CanvasRenderingContext2D

const {PI: pi, sin, cos} = Math

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
    theta: pi - pi / 12 * Math.random(),
    thetaDot: 0,
    phi: 0,
    phiDot: 0
}

const g = 9.81

const m1 = 0.1
const m2 = 0.5
const r = 0.75
const l = 2.0
const J = m2 * r ** 2
const b = 1.5

const params: Params = {
    length: l * 100
}

function getDisturbance(): number {
    if (externalDisturbance.mouseDown) {
        let [x, y] = [externalDisturbance.x, externalDisturbance.y]
        x -= ox
        y -= oy

        const [ex, ey] = [params.length * cos(state.theta), params.length * sin(state.theta)]
        return 0.05 * ((x * ex + y * ey) / Math.sqrt(ex ** 2 + ey ** 2))
    }

    return 0
}

function derivativeWithFriction(state: number[], t: number, dt: number) {
    const [_th, _dth, _phi, _dphi] = state
    const friction = -b * (_dphi - _dth)
    const ddth = (-friction - (0.5 * m1 + m2) * l * g * sin(-_th)) / (m1 * l ** 2 / 3 + m2 * l ** 2 + J)
    const ddphi = friction / J
    return [_dth, ddth, _dphi, ddphi]
}

async function derivativeWithNNControl(state: number[], t: number, dt: number) {
    const [_th, _dth, _phi, _dphi] = state
    const u = (await onnxModelRunner.forward(_th, _dth, _dphi)) + getDisturbance()
    const ddth = (u - (0.5 * m1 + m2) * l * g * sin(-_th)) / (m1 * l ** 2 / 3 + m2 * l ** 2 + J)
    const ddphi = u / J
    return [_dth, ddth, _dphi, ddphi]
}

async function integrate(state: State, t: number, dt: number): Promise<State> {
    const next = await integrateRK4Async([state.theta, state.thetaDot, state.phi, state.phiDot], t, dt, derivativeWithNNControl)
    return {
        theta: next[0],
        thetaDot: next[1],
        phi: next[2],
        phiDot: next[3]
    }
}

function renderDisturbance() {
    ctx.beginPath()
    ctx.save()
    ctx.translate(ox, oy)
    const [ex, ey] = [params.length * cos(state.theta - pi / 2), params.length * sin(state.theta - pi / 2)]
    const [x, y] = [externalDisturbance.x - ox, externalDisturbance.y - oy]
    ctx.translate(ex, ey)
    const d = Math.sqrt((ex - x) ** 2 + (ey - y) ** 2)
    ctx.rotate(Math.atan2(y - ey, x - ex))
    const r2 = 15 + d * 60 / 400
    drawCirclesWithTangentCone(ctx, 15, r2, d, '#BF00008F')

    ctx.restore()
}

async function render() {
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

    if (externalDisturbance.mouseDown) {
        renderDisturbance()
    }

    if (chartU) {
        chartU.data.labels.push('')
        chartU.data.datasets[0].data.push(state.theta)

        if (chartU.data.labels.length > 512) {
            chartU.data.labels.shift()
            chartU.data.datasets[0].data.shift()
        }

        chartU.update('active')
    }

    const now = Date.now()
    const dt = (now - t) / 1000
    state = await integrate(state, t, dt)
    t = now

    requestAnimationFrame(render)
}

window.onload = async () => {
    initUI();
    await onnxModelRunner.init();
    [ox, oy] = [canvas.width / 2, canvas.height / 2];
    await render();
}

const externalDisturbance = (() => {
    let mouseDown = false
    let [x, y] = [0, 0]
    let hasDisturbance = false

    function getScaledCoordinates(event: MouseEvent): [number, number] {
        const rect = canvas.getBoundingClientRect();

        // Calculate mouse position relative to the canvas
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        // If you need to account for CSS scaling of the canvas
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        // Apply scaling if the canvas is scaled by CSS
        const canvasX = x * scaleX;
        const canvasY = y * scaleY;
        return [canvasX, canvasY]
    }

    canvas.addEventListener('mousedown', (e) => {
        mouseDown = true;
        [x, y] = getScaledCoordinates(e)
    })

    canvas.addEventListener('mousemove', (e) => {
        if (mouseDown) {
            [x, y] = getScaledCoordinates(e)
            console.log(x, y)
        }
    })

    canvas.addEventListener('mouseup', (e) => {
        mouseDown = false;
        hasDisturbance = true;
        [x, y] = getScaledCoordinates(e)
    })

    canvas.addEventListener('mouseleave', (e) => {
        mouseDown = false;
        hasDisturbance = true;
        [x, y] = getScaledCoordinates(e)
    })

    return {
        get mouseDown() {
            return mouseDown
        },
        get x() {
            return x
        },
        get y() {
            return y
        }
    }
})()

const onnxModelRunner = (() => {
    let session: ort.InferenceSession
    return {
        init: async () => {
            try {
                session = await ort.InferenceSession.create('./model/fip_solver.onnx');
            } catch (err) {
                console.error("Failed to init inference session", err)
            }
        },
        forward: async (theta: number, thetaDot: number, phiDot: number): Promise<number> => {
            if (session === undefined) {
                throw new Error('Session is not initialized')
            }
            const input = new ort.Tensor('float32', new Float32Array([
                cos(theta),
                sin(theta),
                thetaDot,
                phiDot
            ]), [1, 4])
            const results = await session.run({input})
            return results.scaled_action.data[0] as number
        }
    }
})()

let chartU: Chart | undefined
let chartTheta: Chart | undefined
let chartOmega: Chart | undefined
let chartPhiDot: Chart | undefined

function initUI() {
    const drawer = document.getElementById('drawer');
    const handle = document.getElementById('drawer-handle');
    handle.addEventListener('click', () => drawer.classList.toggle('open'));

    chartU = buildChart('chart-u')
    chartTheta = buildChart('chart-theta')
    chartOmega = buildChart('chart-omega')
    chartPhiDot = buildChart('chart-phi-dot')
}

function buildChart(canvasId: string): Chart {
    const chartUContext = (document.getElementById(canvasId) as HTMLCanvasElement).getContext('2d')

    const initialData = {
        labels: [],
        datasets: [{
            data: [],
            borderColor: 'rgba(0, 0, 0, 0.7)',
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 0,
            pointStyle: false,
        }]
    };

    const config = {
        type: 'line',
        data: initialData,
        options: {
            animation: false,
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: false,
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        display: true,
                        maxTicksLimit: 5,
                        callback: (value: number, _index: any, _values: any) => value.toFixed(1)
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)',
                    },
                    ticks: {
                        display: false,
                        maxTicksLimit: 10
                    }
                }
            },
            backgroundColor: 'transparent'
        }
    };

    return new Chart(chartUContext, config as ChartConfiguration)
}

