"""Verify swing-up, LQR catch, and recovery from seeded random kicks."""

import math
from pathlib import Path

import mujoco
import numpy as np

from controller import SwingUpLQRController, lqr_gain, reduced_state
from disturbance import RandomTangentialKicks


DURATION_S = 18.0
SETTLING_WINDOW_S = 1.0

scene_path = Path(__file__).with_name("scene.xml").resolve()
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)
actuator_id = model.actuator("wheel-torque").id

assert model.nq == 2 and model.nv == 2
assert model.nu == 1

_, state_matrix, input_matrix = lqr_gain(model)
controllability = np.column_stack(
    [
        input_matrix,
        state_matrix @ input_matrix,
        state_matrix @ state_matrix @ input_matrix,
    ]
)
assert np.linalg.matrix_rank(controllability) == 3

mujoco.mj_resetData(model, data)
data.joint("pivot").qpos[0] = math.pi - 1e-3
mujoco.mj_forward(model, data)

controller = SwingUpLQRController(model)
kicks = RandomTangentialKicks(seed=7)
first_catch_time_s: float | None = None
maximum_absolute_wheel_speed = 0.0
maximum_applied_torque = 0.0
maximum_kick_force = 0.0
maximum_post_kick_angle = 0.0
settling_angles: list[float] = []
settling_wheel_speeds: list[float] = []

while data.time < DURATION_S:
    command = controller.update(data)
    if command.mode == "lqr" and first_catch_time_s is None:
        first_catch_time_s = data.time

    data.ctrl[actuator_id] = command.applied_torque_nm
    kick = kicks.apply(model, data, enabled=command.mode == "lqr")
    maximum_kick_force = max(maximum_kick_force, abs(kick.force_n))
    maximum_applied_torque = max(
        maximum_applied_torque, abs(command.applied_torque_nm)
    )

    mujoco.mj_step(model, data)
    state = reduced_state(data)
    maximum_absolute_wheel_speed = max(maximum_absolute_wheel_speed, abs(state[2]))
    if kicks.count > 0:
        maximum_post_kick_angle = max(maximum_post_kick_angle, abs(state[0]))
    if data.time >= DURATION_S - SETTLING_WINDOW_S:
        settling_angles.append(abs(state[0]))
        settling_wheel_speeds.append(abs(state[2]))

assert first_catch_time_s is not None
assert first_catch_time_s < 8.0
assert kicks.count >= 2
assert controller.mode == "lqr"
assert maximum_post_kick_angle <= math.radians(20.0)

settling_angle_deg = math.degrees(max(settling_angles))
settling_wheel_speed_rpm = max(settling_wheel_speeds) * 60.0 / (2.0 * math.pi)
assert settling_angle_deg <= 2.0
assert settling_wheel_speed_rpm <= 100.0

print(f"LQR gain [angle, angle rate, absolute wheel rate]: {controller.gain}")
print(f"pendulum inertia for energy shaping: {controller.inertia:.9f} kg*m^2")
print(f"gravity coefficient: {controller.gravity_coefficient:.9f} N*m")
print(f"first LQR catch: {first_catch_time_s:.3f} s")
print(f"LQR catches: {controller.catch_count}")
print(f"random kicks: {kicks.count}, maximum force: {maximum_kick_force:.3f} N")
print(
    "maximum angle after kicks begin: "
    f"{math.degrees(maximum_post_kick_angle):.3f} deg"
)
print(f"maximum applied motor torque: {maximum_applied_torque:.3f} N*m")
print(f"maximum absolute wheel speed: {maximum_absolute_wheel_speed:.3f} rad/s")
print(
    "last-second angle / wheel-speed bounds: "
    f"{settling_angle_deg:.6f} deg / {settling_wheel_speed_rpm:.6f} rpm"
)
print(f"final mode: {controller.mode}")
print("checks: PASS")
