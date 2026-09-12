# 08 — LQR-controlled flywheel pendulum

This exercise adds an ideal torque actuator to the reaction-wheel hinge and
uses continuous-state LQR feedback to stabilize the pendulum. It begins at rest
10 degrees from upright. Unlike the earlier fixed pulse, torque is recalculated
from the current state before every 2 ms MuJoCo step.

From `software/sim`, run the deterministic five-second recovery check:

```shell
poetry run python learning/mujoco/08-actuated-wheel/check_model.py
```

On macOS, view the controller at half real-time speed:

```shell
poetry run mjpython learning/mujoco/08-actuated-wheel/view.py
```

Press **Backspace** to reset to the 10-degree initial condition.

## Ideal actuator

The MJCF actuator remains deliberately simple:

```xml
<actuator>
  <motor name="wheel-torque" joint="wheel-hinge" gear="1"
         ctrllimited="true" ctrlrange="-1.7 1.7"/>
</actuator>
```

MuJoCo's `motor` is a direct-drive torque source, not an electrical motor model.
The command is limited to the current mj5208 peak-torque assumption of
±1.7 N·m, but there is no torque-speed curve, current loop, command delay, or
battery model yet.

## Controller state

The controller uses the reduced state

```text
x = [pendulum angle, pendulum rate, absolute wheel rate]
```

The wheel hinge reports rotor speed relative to the carrier, so the absolute
wheel rate is

```text
absolute wheel rate = pivot rate + wheel-hinge rate
```

Penalizing absolute wheel rate makes the controller return the wheel toward rest
instead of balancing with steadily accumulating momentum. The wheel's relative
angle is omitted because this axially symmetric model has no preferred rotor
angle.

## LQR design

`controller.py` finite-differences MuJoCo's continuous accelerations around the
motionless upright equilibrium to obtain the reduced linear model

```text
xdot = A x + B u
```

It then solves the continuous algebraic Riccati equation using the same costs as
the Phase 0 controller:

```text
Q = diag(120, 2, 0.015)
R = 0.8
u = clip(-K x, -1.7, +1.7)
```

The gain comes from the linearization, while the rollout itself remains the full
nonlinear MuJoCo model. The check also verifies controllability, recovery from
10 degrees, the last-second angle bound, and wheel despinning.

## Next

The remaining T-001 work is a numerical trajectory comparison between MuJoCo
and the analytical RK4 model for free fall, a fixed diagnostic torque pulse,
and this LQR controller. The pulse no longer drives the viewer; it will exist
only as one comparison scenario.
