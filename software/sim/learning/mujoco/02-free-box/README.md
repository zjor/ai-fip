# 02 — Free box

This exercise moves the box from `worldbody` into its own dynamic `body` and
gives that body a `freejoint`. The box can now translate and rotate in all
three dimensions, fall under gravity, and collide with the floor.

From `software/sim`, run:

```shell
poetry run python learning/mujoco/02-free-box/view.py
```

## Viewer experiments

- Press `Space` to pause and resume the simulation.
- Press `Backspace` to reset the box above the floor.
- Double-click the box to select it.
- Hold `Ctrl` and left-drag the selected box to rotate it.
- Hold `Ctrl` and right-drag to translate it vertically.
- Add `Shift` to the right-drag to translate it horizontally.
- Try perturbing the box while paused and while running.

When paused, perturbation changes the selected body's pose. While running, it
acts through forces and torques, so the body responds according to its mass,
inertia, contacts, and gravity.

## Model experiments

Change one quantity at a time, save `scene.xml`, then press `Ctrl+L`:

- Raise or lower the body's initial `pos="0 0 0.8"`.
- Change the geom's `mass="1"`.
- Make one box dimension longer.
- Change gravity to Moon-like `gravity="0 0 -1.62"`.
- Remove `<freejoint/>` and observe what happens.

The box uses `solref="-1000 -5"` to make its contact with the floor bouncy. The
two negative values select MuJoCo's direct contact format: `1000` is contact
stiffness and `5` is damping. Try these comparisons:

- Remove `solref` to restore the default, nearly non-bouncy contact.
- Use `solref="-1000 0"` for an almost perfectly elastic bounce.
- Increase the damping magnitude, for example `solref="-1000 -10"`, to lose
  more height on every bounce.

These parameters describe the contact solver's response; they do not deform
the rendered box geometry like a soft-body simulation would.

## Things to notice

- A geom directly under `worldbody` is fixed.
- A geom inside a body is still fixed to that body.
- The joint makes the body movable relative to its parent.
- A free joint has six degrees of freedom: three translations and three
  rotations.
- MuJoCo stores its configuration in seven `qpos` values: XYZ position plus a
  four-component unit quaternion. Its velocity uses six `qvel` values.

## Observations

Record anything surprising here as you experiment.
