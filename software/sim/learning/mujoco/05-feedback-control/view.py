import math
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer


TARGET_ANGLE = -math.pi / 2  # upright, rad
INITIAL_ANGLE = TARGET_ANGLE + math.radians(10)
KP = 12.0  # N*m/rad
KD = 3.0  # N*m*s/rad
TORQUE_LIMIT = 3.0  # N*m

DISTURBANCE_TORQUE = 1.5  # N*m
DISTURBANCE_START = 2.0  # simulation seconds
DISTURBANCE_DURATION = 0.1  # simulation seconds
DISTURBANCE_PERIOD = 4.0  # simulation seconds

PLOT_HISTORY = 10.0  # simulation seconds
PLOT_INTERVAL = 0.02  # simulation seconds
PLOT_WIDTH = 900
PLOT_HEIGHT = 440
PLOT_Y_LIMIT = 6.0
GRID_STEP = 1.0

scene_path = Path(__file__).with_name("scene.xml").resolve()
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

motor_id = model.actuator("pivot-motor").id
pivot_dof_id = model.joint("pivot").dofadr[0]

telemetry_figure = mujoco.MjvFigure()
mujoco.mjv_defaultFigure(telemetry_figure)
telemetry_figure.title = "Upright PD control"
telemetry_figure.xlabel = ""
telemetry_figure.flg_legend = 1
telemetry_figure.flg_symmetric = 1
telemetry_figure.flg_extend = 0
telemetry_figure.range[0] = (-PLOT_HISTORY, 0.0)
telemetry_figure.range[1] = (-PLOT_Y_LIMIT, PLOT_Y_LIMIT)
telemetry_figure.gridrgb[:] = telemetry_figure.panergba[:3]
telemetry_figure.linewidth = 2.0

TELEMETRY_LINES = (
    ("motor torque (N*m)", (1.0, 0.45, 0.1)),
    ("angle (rad)", (0.2, 0.65, 1.0)),
    ("angular velocity (rad/s)", (0.3, 0.9, 0.35)),
)
for line_index, (name, color) in enumerate(TELEMETRY_LINES):
    telemetry_figure.linename[line_index] = name
    telemetry_figure.linergb[line_index] = color

sample_capacity = int(PLOT_HISTORY / PLOT_INTERVAL) + 1
telemetry_samples = deque(maxlen=sample_capacity)

TELEMETRY_LINE_COUNT = len(TELEMETRY_LINES)
GRID_FIRST_LINE = TELEMETRY_LINE_COUNT
vertical_grid_capacity = int(PLOT_HISTORY / GRID_STEP) + 1
horizontal_grid_capacity = int(2 * PLOT_Y_LIMIT / GRID_STEP) + 1
for line_index in range(
    GRID_FIRST_LINE,
    GRID_FIRST_LINE + vertical_grid_capacity + horizontal_grid_capacity,
):
    telemetry_figure.linergb[line_index] = (0.18, 0.18, 0.18)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def pd_control(angle: float, angular_velocity: float) -> float:
    """Calculate saturated feedback torque from the current joint state."""
    error = wrap_angle(TARGET_ANGLE - angle)
    unconstrained_torque = KP * error - KD * angular_velocity
    return max(-TORQUE_LIMIT, min(TORQUE_LIMIT, unconstrained_torque))


def periodic_disturbance(simulation_time: float) -> float:
    """Apply alternating external torque pulses after an initial quiet period."""
    if simulation_time < DISTURBANCE_START:
        return 0.0

    elapsed = simulation_time - DISTURBANCE_START
    cycle = int(elapsed // DISTURBANCE_PERIOD)
    if elapsed % DISTURBANCE_PERIOD >= DISTURBANCE_DURATION:
        return 0.0

    direction = 1.0 if cycle % 2 == 0 else -1.0
    return direction * DISTURBANCE_TORQUE


def update_moving_grid(simulation_time: float) -> None:
    """Draw a one-unit grid whose time lines move with simulation time."""
    first_line = GRID_FIRST_LINE
    last_line = first_line + vertical_grid_capacity + horizontal_grid_capacity
    telemetry_figure.linepnt[first_line:last_line] = 0

    grid_time = (
        math.ceil((simulation_time - PLOT_HISTORY) / GRID_STEP) * GRID_STEP
    )
    line_index = first_line
    while grid_time <= simulation_time + 1e-9:
        relative_time = grid_time - simulation_time
        telemetry_figure.linepnt[line_index] = 2
        telemetry_figure.linedata[line_index, :4] = (
            relative_time,
            -PLOT_Y_LIMIT,
            relative_time,
            PLOT_Y_LIMIT,
        )
        line_index += 1
        grid_time += GRID_STEP

    first_horizontal_line = GRID_FIRST_LINE + vertical_grid_capacity
    for offset in range(horizontal_grid_capacity):
        line_index = first_horizontal_line + offset
        y = -PLOT_Y_LIMIT + offset * GRID_STEP
        telemetry_figure.linepnt[line_index] = 2
        telemetry_figure.linedata[line_index, :4] = (
            -PLOT_HISTORY,
            y,
            0.0,
            y,
        )


def update_telemetry_figure(
    simulation_time: float,
    motor_torque: float,
    angle: float,
    angular_velocity: float,
) -> None:
    """Append one telemetry sample and rebuild all three plot lines."""
    telemetry_samples.append(
        (simulation_time, motor_torque, angle, angular_velocity)
    )
    point_count = len(telemetry_samples)
    relative_times = [
        sample[0] - simulation_time for sample in telemetry_samples
    ]

    for line_index in range(TELEMETRY_LINE_COUNT):
        telemetry_figure.linepnt[line_index] = point_count
        telemetry_figure.linedata[line_index, : 2 * point_count : 2] = (
            relative_times
        )
        telemetry_figure.linedata[line_index, 1 : 2 * point_count : 2] = [
            sample[line_index + 1] for sample in telemetry_samples
        ]

    update_moving_grid(simulation_time)


def reset_telemetry() -> None:
    telemetry_samples.clear()
    telemetry_figure.linepnt[:] = 0


def reset_simulation() -> None:
    """Reset MuJoCo and Python state to the near-upright initial condition."""
    mujoco.mj_resetData(model, data)
    data.joint("pivot").qpos[0] = INITIAL_ANGLE
    mujoco.mj_forward(model, data)
    reset_telemetry()


def main() -> None:
    reset_simulation()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        next_log_time = 0.0
        next_plot_time = 0.0
        last_simulation_time = data.time

        while viewer.is_running():
            step_started = time.perf_counter()

            if data.time < last_simulation_time:
                reset_simulation()
                next_log_time = data.time
                next_plot_time = data.time
                print("simulation reset detected; controller and telemetry reset")

            angle = data.joint("pivot").qpos[0]
            angular_velocity = data.joint("pivot").qvel[0]
            disturbance = periodic_disturbance(data.time)

            data.ctrl[motor_id] = pd_control(angle, angular_velocity)
            data.qfrc_applied[pivot_dof_id] = disturbance
            mujoco.mj_step(model, data)

            motor_torque = data.qfrc_actuator[pivot_dof_id]

            if data.time >= next_log_time:
                error = wrap_angle(TARGET_ANGLE - angle)
                print(
                    f"t={data.time:5.2f} s  "
                    f"error={math.degrees(error):+6.2f} deg  "
                    f"motor={motor_torque:+5.2f} N*m  "
                    f"disturbance={disturbance:+4.1f} N*m"
                )
                next_log_time += 0.1

            if data.time >= next_plot_time:
                update_telemetry_figure(
                    data.time,
                    motor_torque,
                    data.joint("pivot").qpos[0],
                    data.joint("pivot").qvel[0],
                )
                scene_viewport = viewer.viewport
                viewer.set_figures(
                    (
                        mujoco.MjrRect(
                            scene_viewport.left,
                            scene_viewport.bottom,
                            min(PLOT_WIDTH, scene_viewport.width),
                            min(PLOT_HEIGHT, scene_viewport.height),
                        ),
                        telemetry_figure,
                    )
                )
                next_plot_time += PLOT_INTERVAL

            last_simulation_time = data.time
            viewer.sync()

            sleep_time = model.opt.timestep - (time.perf_counter() - step_started)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
