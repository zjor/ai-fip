"""Stabilize the reaction-wheel pendulum with saturated LQR feedback."""

import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from controller import control_torque, lqr_gain, reduced_state


INITIAL_ANGLE_DEG = 10.0
LOG_INTERVAL_S = 0.05
PLAYBACK_SPEED = 0.5

scene_path = Path(__file__).with_name("scene.xml").resolve()
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)
actuator_id = model.actuator("wheel-torque").id
gain, _, _ = lqr_gain(model)


def initialize() -> None:
    """Reset to rest, ten degrees away from upright."""
    mujoco.mj_resetData(model, data)
    data.joint("pivot").qpos[0] = math.radians(INITIAL_ANGLE_DEG)
    mujoco.mj_forward(model, data)


def main() -> None:
    initialize()
    print(f"LQR gain [angle, angle rate, absolute wheel rate]: {gain}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        next_log_time = 0.0
        last_simulation_time = data.time

        while viewer.is_running():
            step_started = time.perf_counter()

            if data.time < last_simulation_time:
                initialize()
                next_log_time = 0.0
                print("simulation reset to the 10-degree initial condition")

            requested_torque, applied_torque = control_torque(model, data, gain)
            data.ctrl[actuator_id] = applied_torque
            mujoco.mj_step(model, data)

            if data.time >= next_log_time:
                angle, pivot_rate, absolute_wheel_rate = reduced_state(data)
                print(
                    f"t={data.time:5.2f} s  "
                    f"pendulum={math.degrees(angle):+7.3f} deg  "
                    f"pendulum-rate={pivot_rate:+8.3f} rad/s  "
                    f"wheel-absolute-rate={absolute_wheel_rate:+9.3f} rad/s  "
                    f"torque={applied_torque:+.3f} N*m"
                    + (
                        "  SAT"
                        if not math.isclose(requested_torque, applied_torque)
                        else ""
                    )
                )
                next_log_time += LOG_INTERVAL_S

            last_simulation_time = data.time
            viewer.sync()

            wall_step_duration = model.opt.timestep / PLAYBACK_SPEED
            sleep_time = wall_step_duration - (time.perf_counter() - step_started)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
