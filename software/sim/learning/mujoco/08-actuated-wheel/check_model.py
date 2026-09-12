"""Verify that saturated LQR stabilizes the two-joint MuJoCo model."""

import math
from pathlib import Path

import mujoco
import numpy as np

from controller import control_torque, lqr_gain, reduced_state


INITIAL_ANGLE_DEG = 10.0
DURATION_S = 5.0
SETTLING_WINDOW_S = 1.0

scene_path = Path(__file__).with_name("scene.xml").resolve()
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)
actuator_id = model.actuator("wheel-torque").id

assert model.nq == 2 and model.nv == 2
assert model.nu == 1

gain, state_matrix, input_matrix = lqr_gain(model)
controllability = np.column_stack(
    [
        input_matrix,
        state_matrix @ input_matrix,
        state_matrix @ state_matrix @ input_matrix,
    ]
)
assert np.linalg.matrix_rank(controllability) == 3

mujoco.mj_resetData(model, data)
data.joint("pivot").qpos[0] = math.radians(INITIAL_ANGLE_DEG)
mujoco.mj_forward(model, data)

initial_requested_torque, initial_applied_torque = control_torque(model, data, gain)
assert initial_applied_torque > 0.0, "positive angle needs positive wheel torque"

maximum_angle = abs(reduced_state(data)[0])
maximum_absolute_wheel_speed = 0.0
maximum_applied_torque = 0.0
saturated_steps = 0
settling_angles: list[float] = []
settling_wheel_speeds: list[float] = []

while data.time < DURATION_S:
    requested_torque, applied_torque = control_torque(model, data, gain)
    data.ctrl[actuator_id] = applied_torque
    saturated_steps += int(not math.isclose(requested_torque, applied_torque))
    maximum_applied_torque = max(maximum_applied_torque, abs(applied_torque))

    mujoco.mj_step(model, data)
    state = reduced_state(data)
    maximum_angle = max(maximum_angle, abs(state[0]))
    maximum_absolute_wheel_speed = max(maximum_absolute_wheel_speed, abs(state[2]))
    if data.time >= DURATION_S - SETTLING_WINDOW_S:
        settling_angles.append(abs(state[0]))
        settling_wheel_speeds.append(abs(state[2]))

final_state = reduced_state(data)
settling_angle_deg = math.degrees(max(settling_angles))
settling_wheel_speed_rpm = max(settling_wheel_speeds) * 60.0 / (2.0 * math.pi)

assert maximum_angle <= math.radians(INITIAL_ANGLE_DEG + 0.1)
assert settling_angle_deg <= 0.1
assert settling_wheel_speed_rpm <= 5.0

print(f"LQR gain [angle, angle rate, absolute wheel rate]: {gain}")
print(
    "initial requested / applied torque: "
    f"{initial_requested_torque:+.6f} / {initial_applied_torque:+.6f} N*m"
)
print(f"maximum applied torque: {maximum_applied_torque:.6f} N*m")
print(f"maximum absolute wheel speed: {maximum_absolute_wheel_speed:.6f} rad/s")
print(f"saturated steps: {saturated_steps}")
print(f"maximum pendulum angle: {math.degrees(maximum_angle):.6f} deg")
print(
    "last-second angle / wheel-speed bounds: "
    f"{settling_angle_deg:.6f} deg / {settling_wheel_speed_rpm:.6f} rpm"
)
print(f"final [angle, angle rate, absolute wheel rate]: {final_state}")
print("checks: PASS")
