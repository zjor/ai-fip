# 05 — Feedback control

This exercise closes the loop. Instead of choosing torque from simulation time,
Python reads the hinge angle and angular velocity and calculates motor torque
with a proportional-derivative controller.

From `software/sim`, run:

```shell
poetry run mjpython learning/mujoco/05-feedback-control/view.py
```

The pendulum begins 10 degrees away from the unstable upright equilibrium. The
controller brings it upright and rejects alternating external torque pulses
starting at two seconds.

## Feedback loop

```text
qpos, qvel
    -> wrapped angle error
    -> torque = Kp * error - Kd * qvel
    -> clamp to [-3, 3] N*m
    -> data.ctrl
    -> mj_step
    -> new qpos, qvel
```

The controller is:

```python
error = wrap_angle(TARGET_ANGLE - angle)
unconstrained_torque = KP * error - KD * angular_velocity
torque = max(-TORQUE_LIMIT, min(TORQUE_LIMIT, unconstrained_torque))
```

- The proportional term pushes the angle toward the target.
- The derivative term opposes angular velocity and adds damping.
- Wrapping gives the shortest signed angular error across the `-pi`/`+pi`
  boundary.
- Saturation represents a finite actuator instead of allowing arbitrary torque.

For this joint axis, `-pi/2` rad is upright and `+pi/2` rad is downward. The
initial angle is upright plus 10 degrees. `KP = 12` and `KD = 3` stabilize this
model without reaching the 3 N*m limit from that initial condition.

## Control versus disturbance

The two torque inputs intentionally use different MuJoCo fields:

- `data.ctrl[motor_id]` commands the modeled pivot motor.
- `data.qfrc_applied[pivot_dof_id]` applies an external generalized force. For
  a hinge DOF, this is an external torque in N*m.

The disturbance is not part of the controller. It represents an outside push
that lets us observe recovery.

## Reset behavior

The viewer resets `MjData`, but the exercise detects simulation time moving
backward. It then restores the near-upright initial condition and clears the
Python-owned telemetry buffer. The controller itself has no memory yet; a
future integral controller or state estimator would also need an explicit
reset here.

## Experiments

Change one thing at a time:

- Set `KD = 0` and observe undamped or growing oscillations.
- Set `KP = 0` and observe why velocity damping alone cannot select upright.
- Increase and decrease each gain independently.
- Reduce `TORQUE_LIMIT` and find the largest initial error that still recovers.
- Increase `INITIAL_ANGLE` from 10 to 20 or 30 degrees.
- Increase `DISTURBANCE_TORQUE` or its duration until recovery fails.
- Set `TARGET_ANGLE = math.pi / 2` and choose an initial angle near the stable
  downward equilibrium.

This actuator acts directly at the pendulum pivot. It teaches feedback and
saturation, but it is not yet reaction-wheel control. The next model adds a
second rotating body and obtains pendulum torque only through their coupled
dynamics.
