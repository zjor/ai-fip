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

## Next

The remaining T-001 work is a numerical trajectory comparison between MuJoCo
and the analytical RK4 model for free fall, a diagnostic fixed torque pulse,
and the LQR trajectory. Swing-up and kick recovery are now available as extra
scenarios for the later honest-model validation.
