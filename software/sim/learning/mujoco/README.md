# MuJoCo learning exercises

Small, cumulative exercises for learning MuJoCo while working toward
[T-001](../../../../docs/project/tasks.md). Each numbered directory is a runnable
snapshot and should remain understandable on its own.

Assets reused by multiple exercises live in [`assets/`](assets/). Exercise MJCF
files reference that directory with paths relative to their own location, so a
mesh has one canonical copy without depending on another numbered exercise.

Run commands from `software/sim` so every exercise uses this subproject's
Poetry environment.

## Exercises

1. [`01-static-box`](01-static-box/) — load a minimal MJCF scene and explore
   the interactive viewer and camera controls.
2. [`02-free-box`](02-free-box/) — add a dynamic body and observe gravity,
   contact, free-joint state, and mouse perturbations.
3. [`03-pendulum`](03-pendulum/) — constrain a compound body to one hinge and
   explore joint axes, local coordinates, inertia, and damping.
4. [`04-periodic-kicks`](04-periodic-kicks/) — step the model from Python and
   apply periodic torque pulses through a hinge motor actuator.
5. [`05-feedback-control`](05-feedback-control/) — close the loop with a
   saturated PD controller and reject external torque disturbances near upright.
6. [`06-cad-wheel`](06-cad-wheel/) — import the Onshape reaction-wheel STL and
   compare its compiled mass properties with a same-envelope solid cylinder.
7. [`07-flywheel-pendulum`](07-flywheel-pendulum/) — assemble the passive
   flywheel pendulum with a world-fixed support, motor housing and independently
   hinged CAD wheel, without actuation or contacts.
8. [`08-actuated-wheel`](08-actuated-wheel/) — swing up with energy shaping,
   catch and despin with LQR, then reject seeded random force pulses applied
   perpendicular to the rod.

Add later exercises as new numbered directories instead of rewriting earlier
ones. Keep observations that are specific to an exercise in that exercise's
README. Project task state remains in `docs/project/tasks.md`.
