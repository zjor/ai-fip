import {drawPieCircle, drawArmLink} from "./geometry";

const canvas = document.getElementById('canvas') as HTMLCanvasElement
const ctx = canvas.getContext('2d') as CanvasRenderingContext2D

const {PI: pi} = Math

let [ox, oy] = [0, 0]
let [x, y] = [0, 0]
let [dx, dy] = [Math.random() * 5, Math.random() * 5]

function render() {
    ctx.fillStyle = "#fff"
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    ctx.save()
    ctx.translate(ox, oy)
    ctx.rotate(-pi / 3)
    drawArmLink(ctx, 60, 50, 250, 2, '#000', '#fff')
    drawPieCircle(ctx, 20, 2, '#000')
    ctx.restore()

    requestAnimationFrame(render)
}


window.onload = () => {
    [ox, oy] = [canvas.width / 2, canvas.height / 2]
    render()
}

