"""Check the passive world's structure and its basic coupled motion."""

import math
from pathlib import Path

import mujoco
import numpy as np


scene_path = Path(__file__).with_name("scene.xml").resolve()
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

pendulum_id = model.body("pendulum").id
wheel_id = model.body("wheel").id
post_id = model.geom("support-post").id
axle_id = model.geom("support-axle").id

assert model.nq == 2, f"expected two positions, got {model.nq}"
assert model.nv == 2, f"expected two velocities, got {model.nv}"
assert model.body_parentid[wheel_id] == pendulum_id
np.testing.assert_allclose(model.joint("pivot").axis, (0, 1, 0))
np.testing.assert_allclose(model.joint("wheel-hinge").axis, (0, 1, 0))

post_top = model.geom_pos[post_id, 2] + model.geom_size[post_id, 2]
axle_top = model.geom_pos[axle_id, 2] + model.geom_size[axle_id, 0]
assert post_top >= axle_top, "support post does not contain the full axle diameter"

mass_matrix = np.zeros((model.nv, model.nv))
mujoco.mj_forward(model, data)
mujoco.mj_fullM(model, data, mass_matrix)
np.testing.assert_allclose(mass_matrix, mass_matrix.T)
assert np.all(np.linalg.eigvalsh(mass_matrix) > 0), "mass matrix is not positive definite"

data.joint("pivot").qpos[0] = math.radians(10)
mujoco.mj_forward(model, data)
initial_angle = data.joint("pivot").qpos[0]

while data.time < 0.25:
    mujoco.mj_step(model, data)

pendulum_angle = data.joint("pivot").qpos[0]
pendulum_speed = data.joint("pivot").qvel[0]
relative_wheel_speed = data.joint("wheel-hinge").qvel[0]
absolute_wheel_speed = pendulum_speed + relative_wheel_speed

assert pendulum_angle > initial_angle, "upright pendulum did not begin to fall"
assert relative_wheel_speed * pendulum_speed < 0, "wheel did not counter-rotate"
assert absolute_wheel_speed * pendulum_speed > 0, "bearing friction did not drag the wheel"
assert abs(absolute_wheel_speed) < abs(pendulum_speed), "wheel did not slip at its hinge"

print("structure: two nested Y-axis hinge joints")
print(f"carrier mass: {model.body_mass[pendulum_id]:.6f} kg")
print(f"wheel mass: {model.body_mass[wheel_id]:.6f} kg")
print(f"support top / axle top: {post_top:.3f} / {axle_top:.3f} m")
print("upright generalized mass matrix (kg*m^2):")
print(mass_matrix)
print(f"angle after 0.25 s: {math.degrees(pendulum_angle):.3f} deg")
print(f"pendulum rate: {pendulum_speed:.6f} rad/s")
print(f"relative wheel rate: {relative_wheel_speed:.6f} rad/s")
print(f"absolute wheel rate: {absolute_wheel_speed:.3e} rad/s")
print("checks: PASS")
