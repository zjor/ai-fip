# MuJoCo learning exercises

Small, cumulative exercises for learning MuJoCo while working toward
[T-001](../../../../docs/project/tasks.md). Each numbered directory is a runnable
snapshot and should remain understandable on its own.

Run commands from `software/sim` so every exercise uses this subproject's
Poetry environment.

## Exercises

1. [`01-static-box`](01-static-box/) — load a minimal MJCF scene and explore
   the interactive viewer and camera controls.
2. [`02-free-box`](02-free-box/) — add a dynamic body and observe gravity,
   contact, free-joint state, and mouse perturbations.
3. [`03-pendulum`](03-pendulum/) — constrain a compound body to one hinge and
   explore joint axes, local coordinates, inertia, and damping.

Add later exercises as new numbered directories instead of rewriting earlier
ones. Keep observations that are specific to an exercise in that exercise's
README. Project task state remains in `docs/project/tasks.md`.
