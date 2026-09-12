"""Run the passive flywheel pendulum."""

import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer


INITIAL_PENDULUM_ANGLE = math.radians(10.0)
LOG_INTERVAL = 0.1
PLAYBACK_SPEED = 0.5  # simulated seconds per wall-clock second

scene_path = Path(__file__).with_name("scene.xml").resolve()
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

pivot_id = model.joint("pivot").id
wheel_hinge_id = model.joint("wheel-hinge").id


def initialize() -> None:
    """Reset to rest, ten degrees away from the upright equilibrium."""
    mujoco.mj_resetData(model, data)
    data.joint("pivot").qpos[0] = INITIAL_PENDULUM_ANGLE
    data.joint("wheel-hinge").qpos[0] = 0.0
    mujoco.mj_forward(model, data)


def main() -> None:
    initialize()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        next_log_time = 0.0
        last_simulation_time = data.time

        while viewer.is_running():
            step_started = time.perf_counter()

            if data.time < last_simulation_time:
                initialize()
                next_log_time = 0.0
                print("simulation reset to the 10-degree initial condition")

            mujoco.mj_step(model, data)

            if data.time >= next_log_time:
                pendulum_angle = data.joint("pivot").qpos[0]
                pendulum_speed = data.joint("pivot").qvel[0]
                relative_wheel_angle = data.joint("wheel-hinge").qpos[0]
                relative_wheel_speed = data.joint("wheel-hinge").qvel[0]
                absolute_wheel_speed = pendulum_speed + relative_wheel_speed
                print(
                    f"t={data.time:5.2f} s  "
                    f"pendulum={math.degrees(pendulum_angle):+7.2f} deg  "
                    f"wheel-relative={math.degrees(relative_wheel_angle):+8.2f} deg  "
                    f"pendulum-rate={pendulum_speed:+7.3f} rad/s  "
                    f"wheel-absolute-rate={absolute_wheel_speed:+9.6f} rad/s"
                )
                next_log_time += LOG_INTERVAL

            last_simulation_time = data.time
            viewer.sync()

            wall_step_duration = model.opt.timestep / PLAYBACK_SPEED
            sleep_time = wall_step_duration - (time.perf_counter() - step_started)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
