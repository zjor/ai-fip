# 06 — CAD reaction wheel

This exercise imports a binary STL exported from Onshape, verifies its scale
and orientation, and compares its compiled inertia with a same-mass solid
cylinder occupying the same outer envelope.

The STL format does not store units. The source coordinates are millimetres,
so the MJCF mesh asset applies `scale="0.001 0.001 0.001"`. The wheel's axle is
the source X axis. The mesh is closed and centred at its origin. Its canonical
copy is [`../assets/wheel.stl`](../assets/wheel.stl), shared with later
exercises instead of duplicated in each directory.

From `software/sim`, inspect the compiled model without opening a GUI:

```shell
poetry run python learning/mujoco/06-cad-wheel/inspect_model.py
```

Open the scene with:

```shell
poetry run python learning/mujoco/06-cad-wheel/view.py
```

The scene keeps the wheel fixed and disables contacts. This separates mesh
import, scaling and rendering from joints and collision behavior. The next
exercise can reuse the mesh as visual geometry while giving the reaction wheel
a hinge and deliberately selected collision and inertial properties.

## Mass and inertia assumptions

The mesh geom uses a PLA density of 1250 kg/m³. MuJoCo computes exact volume
integrals over the closed mesh because the asset specifies `inertia="exact"`.
This is useful for checking the CAD export, but the physical model should
eventually use the measured mass and inertia of the printed wheel plus bolts,
rotor and fasteners.

The `polished-metal` material changes only rendering. In MuJoCo's classic
viewer, a cool-grey color with high `specular` and `shininess` produces the
metallic appearance; it does not change the geom's PLA density or inertia.

The comparison cylinder has the mesh's mass, outer radius and thickness. It is
not a collision approximation: a solid cylinder fills the wheel's holes and
therefore distributes mass differently. Mesh collision would likewise use a
convex representation rather than reproduce every opening, so the imported
mesh is visual-only in this exercise.
