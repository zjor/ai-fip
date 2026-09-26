# 08 — Swing-up, LQR catch, and random kicks

This exercise controls the complete motion in one simulation. It starts just
beside the downward equilibrium, pumps pendulum energy through the reaction
wheel, hands over to LQR near upright, and then applies random external pushes
to test disturbance recovery. Torque is recalculated before every 2 ms MuJoCo
step.

From `software/sim`, run the deterministic 18-second scenario:

```shell
poetry run python learning/mujoco/08-actuated-wheel/check_model.py
```

On macOS, view it at half real-time speed:

```shell
poetry run mjpython learning/mujoco/08-actuated-wheel/view.py
```

Press **Backspace** to restart near the hanging position. Controller mode
changes and kick events are printed in the terminal.

Compare the same model against an analytical RK4 integrator:

```shell
poetry run python learning/mujoco/08-actuated-wheel/compare_rk4.py
```

## Viewer plots

Four synchronized plots are stacked from the rendering viewport's top-right:

1. requested control `u`, before torque limiting;
2. absolute wheel angular velocity;
3. actual motor torque after limiting;
4. wrapped rod angle in degrees, with zero at upright.

Their compact titles are `u`, `ω`, `τ`, and `θ`. Each panel is now composed from
two independent pixel-space components: a 54 px header and a chart body. The
header renders DejaVu Sans with 14 px left and top padding. The requested panel
size is 576 × 390 px and all panels shrink identically only when the viewport
cannot fit the stack.

The chart body uses a fixed 72 px scale gutter and fixed frame coordinates.
Only the mapping from telemetry values to pixels changes when the Y range
changes; header, gutter, frame width, and frame height do not move. Rendering
the panels directly as images avoids `MjvFigure`'s data-dependent tick layout.

All plots use a rolling time axis from −10 s to 0, where zero is the newest
sample. Each vertical range follows its visible signal and adds 20% of the data
span on both sides; a small minimum range keeps constant signals readable. Grid
lines are hidden. Resetting the simulation also clears all plot history.

## Ideal actuator

The wheel hinge has a direct-drive torque source limited to the current mj5208
peak-torque assumption:

```xml
<motor name="wheel-torque" joint="wheel-hinge" gear="1"
       ctrllimited="true" ctrlrange="-1.7 1.7"/>
```

This is not yet an electrical motor model: there is no torque-speed curve,
current loop, command delay, or battery model. During swing-up, software applies
a lower ±0.5 N·m cap to avoid needlessly accumulating wheel momentum.

## State and LQR

The reduced controller state is

```text
x = [pendulum angle, pendulum rate, absolute wheel rate]
```

The hinge reports wheel speed relative to the carrier, so

```text
absolute wheel rate = pivot rate + wheel-hinge rate
```

`controller.py` finite-differences MuJoCo's continuous accelerations around the
motionless upright equilibrium to obtain `A` and `B`. It solves the continuous
Riccati equation with the Phase 0 costs:

```text
Q = diag(120, 2, 0.015)
R = 0.8
u_lqr = clip(-K x, -1.7, +1.7)
```

Penalizing absolute wheel rate makes LQR despin the wheel instead of balancing
with steadily accumulating momentum. Relative wheel angle is omitted because
the axially symmetric wheel has no preferred rotor angle.

## Swing-up and catch

With angle zero at upright and ±π at hanging, the pendulum energy relative to
upright is

```text
E = 0.5 I theta_rate² + G (cos(theta) - 1)
```

The controller derives `I` and `G` from the compiled MuJoCo model. While energy
is below the upright target it commands

```text
u_swing = clip(-k (-E) theta_rate, -0.5, +0.5)
```

Motor torque acts oppositely on the carrier, hence the minus sign. The controller
switches to LQR inside a 30° catch cone when the energy error is within 5% of
the hanging-to-upright energy difference. If it leaves 45°, it falls back to
swing-up mode.

## Random perpendicular kicks

After the first LQR catch, `disturbance.py` schedules seeded random pushes every
2.5–3.5 seconds, following an initial 1.0–1.5 second delay. Each push lasts
40 ms and has a random sign and magnitude from 2–4 N.

The force is applied at the wheel centre with `mujoco.mj_applyFT`. Its world
direction is the pendulum body's local X axis, while the rod lies on local Z;
those axes remain perpendicular as the pendulum rotates. This models an actual
lateral poke rather than injecting torque through the motor actuator. Seed 7
makes checks and viewer resets reproducible.

## Analytical RK4 comparison

`rk4.py` uses the reduced state `x = [θ, θ̇, ω]`, where `ω` is absolute wheel
speed. For applied wheel torque `u`, carrier inertia `I`, axial wheel inertia
`J`, gravity coefficient `G`, pivot damping `b_p` and relative wheel-hinge
damping `b_w`, its equations are:

```text
τ_w = u − b_w (ω − θ̇)
θ̈ = (G sin θ − b_p θ̇ − τ_w) / I
ω̇ = τ_w / J
```

The constants come from the compiled `scene.xml` mass matrix, gravity bias at
90°, and joint damping. For the current scene they are `I = 0.028399817 kg·m²`,
`J = 0.001464308 kg·m²`, `G = 1.134974359 N·m`, `b_p = 0.0002 N·m·s/rad`, and
`b_w = 0.005 N·m·s/rad`. This avoids a second, drifting set of physical inputs
while keeping the differential equations and RK4 step independent of MuJoCo's
dynamics solver. Torque is held constant over each 2 ms step. LQR uses the same
gain but computes its own torque from each model's state, with ±1.7 N·m clipping.

`compare_rk4.py` checks free fall from 10° for 0.7 s, a +0.2 N·m pulse for
0.1 s from upright followed by 0.4 s of coasting, and LQR from 5° for 2 s.
The maximum trajectory errors must stay below these limits:

| MuJoCo integrator | Scenario | Angle | Pivot rate | Wheel rate | Torque |
|---|---|---:|---:|---:|---:|
| Euler (scene default) | Free fall | 0.8° | 0.02 rad/s | 0.05 rad/s | 0 N·m |
| Euler (scene default) | Fixed pulse | 0.2° | 0.02 rad/s | 0.05 rad/s | 0 N·m |
| Euler (scene default) | LQR | 0.2° | 0.02 rad/s | 0.2 rad/s | 0.02 N·m |
| RK4 | Each scenario | 0.00001° | 0.00001 rad/s | 0.00001 rad/s | 0.00001 N·m |

The Euler bounds cover integration error at the scene's 2 ms step. In the
current comparison, the largest Euler angle error is 0.68834° during free fall;
the largest LQR wheel-rate error is 0.15936 rad/s. Switching MuJoCo to RK4 for
the same checks reduces all printed differences below 0.00000001 in their
respective units. The runner executes both integrators and exits nonzero if a
bound is exceeded. Angle differences are wrapped to ±π before comparison. The
RK4 result checks the equations, parameter extraction, actuation signs and
state mapping; the Euler result bounds the numerical
discrepancy of the scene as normally run. Swing-up and kick recovery remain
separate scenarios for later honest-model validation.
