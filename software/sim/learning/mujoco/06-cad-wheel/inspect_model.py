"""Print compiled CAD-wheel geometry and compare it with a solid cylinder."""

from pathlib import Path

import mujoco
import numpy as np


scene_path = Path(__file__).with_name("scene.xml").resolve()
model = mujoco.MjModel.from_xml_path(str(scene_path))

mesh_id = model.mesh("wheel-mesh").id
vertex_start = model.mesh_vertadr[mesh_id]
vertex_count = model.mesh_vertnum[mesh_id]
vertices = model.mesh_vert[vertex_start : vertex_start + vertex_count]
dimensions = np.ptp(vertices, axis=0)

wheel_id = model.body("wheel").id
mass = model.body_mass[wheel_id]
principal_inertia = model.body_inertia[wheel_id]

# The source mesh is axial along X and has equal Y/Z envelope dimensions.
thickness = dimensions[0]
radius = max(dimensions[1:]) / 2
solid_cylinder_axial_inertia = 0.5 * mass * radius**2
mesh_axial_inertia = max(principal_inertia)

print(f"vertices: {vertex_count}")
print(f"dimensions: {dimensions * 1e3} mm")
print(f"mass at 1250 kg/m^3: {mass * 1e3:.3f} g")
print(f"principal inertia: {principal_inertia} kg*m^2")
print(f"mesh axial inertia: {mesh_axial_inertia:.9f} kg*m^2")
print(
    "same-mass solid-cylinder axial inertia: "
    f"{solid_cylinder_axial_inertia:.9f} kg*m^2"
)
print(
    "mesh / solid-cylinder axial inertia: "
    f"{mesh_axial_inertia / solid_cylinder_axial_inertia:.3f}"
)
print(f"comparison cylinder radius: {radius * 1e3:.3f} mm")
print(f"comparison cylinder thickness: {thickness * 1e3:.3f} mm")
