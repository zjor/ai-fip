# 04 — Periodic kicks from Python

This exercise replaces the viewer-owned simulation loop with a Python-owned
loop. The XML defines a motor on the pendulum hinge; Python writes its torque
command before every physics step.

From `software/sim`, run:

```shell
poetry run mjpython learning/mujoco/04-periodic-kicks/view.py
```

On macOS, MuJoCo's passive viewer must run through `mjpython` so its GUI stays
on the main thread. The earlier viewer-owned exercises can use ordinary
`python`.

The motor applies a `+3 N*m` pulse for 0.15 simulation seconds every two
seconds. The pulse direction alternates between positive and negative. A
large graph overlays the last ten seconds of applied joint torque, pendulum
angle, and angular velocity on the rendered scene. Its legend identifies each
line and includes the corresponding unit. A faint grid has one-second spacing
along the X-axis and one-unit spacing along the Y-axis.

## The control path

```text
periodic_kick(data.time)
        -> data.ctrl[motor_id]
        -> mujoco.mj_step(model, data)
        -> new data.qpos and data.qvel
        -> viewer.sync()
```

- `MjModel` contains the compiled, mostly constant model: bodies, joints,
  actuators, masses, limits, and timestep.
- `MjData` contains the changing simulation state: time, positions,
  velocities, forces, sensor values, and actuator commands.
- `<motor joint="pivot" gear="1">` maps one scalar control value to torque
  around the hinge axis. With `gear="1"`, a control value of `3` means
  `3 N*m`.
- `data.ctrl[...]` is the actuator input vector. Set it before `mj_step`.
- `mj_step` advances the physics by `model.opt.timestep`, which is 0.002 s in
  this scene.
- `data.joint("pivot").qpos[0]` and `.qvel[0]` read the hinge angle in radians
  and angular velocity in radians per second.
- `data.qfrc_actuator[pivot_dof_id]` is the generalized actuator force at the
  hinge. Because a hinge coordinate is angular, this value is torque in N*m.
- `MjvFigure` holds graph data, and `viewer.set_figures(...)` overlays it in a
  pixel rectangle on the viewer.

## Figure placement

Each graph is paired with a viewport rectangle:

```python
mujoco.MjrRect(left, bottom, width, height)
```

The four values are window-relative pixels, while `viewer.viewport` describes
the scene rectangle remaining after MuJoCo lays out its sidebars. Its `left`
and `bottom` fields are therefore essential offsets; its `width` and `height`
give the usable scene size. The graph is anchored to the scene's bottom-left
corner as:

```python
mujoco.MjrRect(
    viewport.left,
    viewport.bottom,
    plot_width,
    plot_height,
)
```

The requested size is 900 x 440 pixels. Width and height are capped to the
actual scene dimensions so the graph remains inside the viewport. Graph
overlays are visual only: MuJoCo does not treat them as UI widgets that consume
mouse events. Keeping the graph inside the scene prevents clicks from reaching
sidebar controls, although a click can still reach the underlying 3D scene.

MuJoCo's native figure grid is fixed inside the axes, so it cannot move with a
scrolling signal. The script hides that grid and draws faint grid lines as
additional unnamed figure lines. Each vertical line is placed at
`integer_time - data.time`, making it travel left continuously until it leaves
the ten-second window. Horizontal lines remain fixed one unit apart.

The Y range is `[-6, 6]`, which necessarily makes 12 one-unit cells. The time
window is ten seconds wide and therefore has ten one-second cells, including
partial cells moving through its left and right boundaries.

## Reset handling

The viewer's Reset command resets `MjData`, including `data.time`, but it does
not reset Python variables. The loop detects `data.time` moving backward,
clears the telemetry buffer, and restarts the plot and log schedules at the new
time. The torque sequence also restarts because `periodic_kick` uses
`data.time`.

The script logs `angle` and `angular_velocity` every 0.1 simulation seconds.
They are observations only: the pulse currently depends on time, not state. In
the next control exercise, these values can become feedback inputs.

## Experiments

- Change `KICK_TORQUE`, `KICK_DURATION`, and `KICK_PERIOD` independently.
- Return `KICK_TORQUE * math.sin(2 * math.pi * simulation_time)` from the
  controller to apply a smooth periodic torque instead of pulses.
- Replace `periodic_kick(data.time)` with `0.0` and recover the passive
  pendulum from exercise 03.
- Change `ctrlrange` in XML and command a larger value. MuJoCo clips the motor
  command because `ctrllimited="true"`.
- Change the logging interval. Printing every 0.002 s will make the viewer
  unnecessarily slow.
- Toggle individual lines by commenting out their `linepnt` and `linedata`
  updates. Each first index in `linedata` represents one graph line.

## Torque versus external force

Because this model has a hinge, a motor torque is its natural control input.
For a temporary world-space push on a body, MuJoCo also provides
`data.xfrc_applied[body_id]`, whose six values are force XYZ followed by torque
XYZ. That is useful for disturbances, while `data.ctrl` represents the
actuator that a controller will eventually command.
