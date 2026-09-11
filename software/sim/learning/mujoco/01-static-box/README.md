# 01 — Static box

This exercise introduces the smallest useful MJCF scene and the standalone
MuJoCo viewer. The box is attached directly to `worldbody`, so it is fixed in
the world and has no joint or dynamic state.

From `software/sim`, open it with the exercise launcher:

```shell
poetry run python learning/mujoco/01-static-box/view.py
```

Or, from the repository root, use:

```shell
poetry --directory software/sim run python learning/mujoco/01-static-box/view.py
```

The launcher constructs an absolute path relative to `view.py`, so it does not
depend on the shell's current working directory.

When using the viewer directly, `--mjcf` is resolved relative to the Python
process's working directory. You can inspect that directory and test the path
without opening the viewer:

```shell
poetry run python -c 'from pathlib import Path; p = Path("learning/mujoco/01-static-box/scene.xml"); print(Path.cwd()); print(p.resolve()); print(p.is_file())'
```

## Viewer experiments

- Left-drag to orbit the camera around the scene.
- Right-drag to pan vertically; Shift-right-drag pans horizontally.
- Scroll, or middle-drag, to zoom.
- Right-double-click a point to center the camera there.
- Press `F1` to show the viewer's help overlay.
- Edit `scene.xml`, save it, then press `Ctrl+L` in the viewer to reload it.

Try changing `pos`, `size`, and `rgba` one at a time. In MuJoCo, a box's three
`size` values are half-sizes, so `size="0.3 0.2 0.1"` produces a box measuring
0.6 × 0.4 × 0.2 m.

The checkerboard floor demonstrates MuJoCo's three-step texture relationship:

1. `<texture>` generates the checker image.
2. `<material>` controls how the image repeats and reflects light.
3. The floor `<geom>` refers to that material by name.

Change `rgb1` and `rgb2` to recolor the squares. Change the material's
`texrepeat` to control their density; because `texuniform="true"`, repetition
is measured consistently in world-space units.

## Things to notice

- `worldbody` is the root of the kinematic tree.
- A `geom` supplies visible and collision geometry.
- A geom placed directly in `worldbody` is static.
- Orbiting changes the camera, not the box pose.

## Observations

Record anything surprising here as you experiment.
