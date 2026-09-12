# 07 — Flywheel pendulum

This exercise builds the passive world for the reaction-wheel pendulum. A
world-fixed post and axle hold a pendulum carrier. The carrier has one pivot
hinge, and the CAD wheel is a child body with a second hinge on the same Y axis.
There is no actuator or controller yet.

From `software/sim`, run the deterministic structural and motion check:

```shell
poetry run python learning/mujoco/07-flywheel-pendulum/check_model.py
```

On macOS, open the live simulation with:

```shell
poetry run mjpython learning/mujoco/07-flywheel-pendulum/view.py
```

The pendulum begins at rest 10 degrees away from upright. Gravity makes it
fall. The wheel counter-rotates relative to the beam, while slight bearing
friction drags it in the direction of the beam's absolute rotation.

Press **Space** to pause or resume and **Backspace** to reset to the initial
condition. The terminal reports the pendulum angle, relative wheel angle,
pendulum rate, and absolute wheel rate.

`PLAYBACK_SPEED = 0.5` in `view.py` displays one simulated second over two
wall-clock seconds. This changes only the viewer pacing; the model still uses
its 2 ms physics timestep and produces the same simulated trajectory.

## Model structure

```text
world
├── floor, support post and support axle (fixed geoms)
└── pendulum body
    ├── pivot hinge
    ├── beam geom
    ├── motor-housing cylinder
    └── wheel body
        ├── wheel hinge
        └── CAD wheel mesh
```

MuJoCo joints connect bodies, not geoms. The support axle and motor cylinder
make the mechanism readable visually, but they do not themselves create the
joints. The motor housing belongs rigidly to the pendulum; the wheel hinge
provides rotor motion relative to it.

Both dynamic geoms have contacts disabled. The hinges use provisional
viscous damping values: 0.0002 N·m·s/rad at the carrier pivot and
0.005 N·m·s/rad at the wheel bearing. These are learning values rather than
measured hardware properties. The 450 mm pivot height gives the
250 mm carrier and 92.8 mm wheel enough clearance to pass through the hanging
position. The support post reaches 480 mm, 30 mm above the axle centre, so it
contains the complete 20 mm axle diameter rather than meeting its centreline.
The floor and support remain useful visual references.

## Provisional mass model

- beam: 0.064 kg, matching the current CAD estimate;
- motor housing: 0.193 kg, the catalogue mass of the mj5208;
- prototype wheel: inferred from its exact closed mesh at 1250 kg/m³ PLA,
  approximately 0.238 kg.

These values make the two-body dynamics meaningful, but they are not the final
hardware partition. Rotor mass, fasteners, tuning bolts, electronics and the
final wheel geometry will be assigned explicitly after the basic two-joint
physics and sign conventions are validated.

At the upright pose, the resulting generalized pivot inertia is approximately
0.02986 kg·m², close to the current Phase 0 nominal value of 0.02887 kg·m².
The prototype wheel's axial inertia remains lower than the final design target.

## Coordinates and state

- Z is up and both hinge axes are +Y, so motion stays in the X-Z plane.
- `pivot = 0` is the unstable upright pose.
- `wheel-hinge` is the wheel angle relative to the pendulum.
- relative wheel speed is the motor's rotor-to-stator speed;
- absolute wheel speed is `pivot rate + wheel-hinge rate`.

The next exercise can add an ideal torque actuator to `wheel-hinge` and verify
equal-and-opposite angular acceleration before any feedback controller is used.
