# 03 — Pendulum

This exercise replaces the six-degree-of-freedom joint with a hinge. A rod and
a bob belong to the same body, so they move as one rigid pendulum around the
pivot.

From `software/sim`, run:

```shell
poetry run python learning/mujoco/03-pendulum/view.py
```

The pendulum starts horizontally. Gravity creates torque around the hinge, so
it falls and swings past its lowest point. Joint damping gradually removes
energy.

## Viewer experiments

- Press `Space` to pause and resume.
- Press `Backspace` to release the pendulum from its initial pose again.
- Double-click the rod or bob to select the pendulum body.
- Apply mouse perturbations and notice that the hinge rejects motion outside
  its allowed axis.
- Expand the right-side **Joint** panel and watch the `pivot` coordinate.
- Under **Rendering**, set **Label** to `Joint` to locate the hinge.

## Model experiments

Change one quantity at a time, save `scene.xml`, then press `Ctrl+L`:

- Set `damping="0"` and observe how long the pendulum continues swinging.
- Increase damping to `0.1` and compare how quickly it settles.
- Change the hinge axis from `0 1 0` to `1 0 0` and observe the new plane of
  allowed motion.
- Try `axis="0 0 1"` and explain why gravity no longer starts the motion.
- Change the rod and bob endpoint from `0.7` to `1.0` and compare the period.
- Change the bob's mass and predict which aspects of the motion should change.

## Things to notice

- `body pos="0 0 1.2"` places the body's local frame—and therefore the
  default joint position—at the pivot.
- `fromto` and the bob's `pos` are expressed in that body's local frame.
- The hinge axis `0 1 0` is the local Y axis, so the pendulum moves in the XZ
  plane.
- Both geoms are rigidly attached to one body. They do not need a joint between
  them.
- MuJoCo infers the body's center of mass and inertia from the two geom shapes
  and masses.
- A hinge contributes one value to `qpos` and one value to `qvel`.

## Observations

Record predictions and results here as you experiment.
