import math
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer


KICK_TORQUE = 3.0  # N*m
KICK_DURATION = 0.15  # simulation seconds
KICK_PERIOD = 2.0  # simulation seconds
PLOT_HISTORY = 10.0  # simulation seconds
PLOT_INTERVAL = 0.02  # simulation seconds
PLOT_WIDTH = 900
PLOT_HEIGHT = 440
PLOT_Y_LIMIT = 6.0
GRID_STEP = 1.0

scene_path = Path(__file__).with_name("scene.xml").resolve()
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

motor_id = model.actuator("kick-motor").id
pivot_dof_id = model.joint("pivot").dofadr[0]


telemetry_figure = mujoco.MjvFigure()
mujoco.mjv_defaultFigure(telemetry_figure)
telemetry_figure.title = "Pendulum telemetry"
telemetry_figure.xlabel = ""
telemetry_figure.flg_legend = 1
telemetry_figure.flg_symmetric = 1
telemetry_figure.flg_extend = 0
telemetry_figure.range[0] = (-PLOT_HISTORY, 0.0)
telemetry_figure.range[1] = (-PLOT_Y_LIMIT, PLOT_Y_LIMIT)
# Hide MuJoCo's fixed grid. Moving grid lines are added below as plot lines.
telemetry_figure.gridrgb[:] = telemetry_figure.panergba[:3]
telemetry_figure.linewidth = 2.0

telemetry_figure.linename[0] = "torque (N*m)"
telemetry_figure.linergb[0] = (1.0, 0.45, 0.1)
telemetry_figure.linename[1] = "angle (rad)"
telemetry_figure.linergb[1] = (0.2, 0.65, 1.0)
telemetry_figure.linename[2] = "angular velocity (rad/s)"
telemetry_figure.linergb[2] = (0.3, 0.9, 0.35)

sample_capacity = int(PLOT_HISTORY / PLOT_INTERVAL) + 1
telemetry_samples = deque(maxlen=sample_capacity)

TELEMETRY_LINE_COUNT = 3
GRID_FIRST_LINE = TELEMETRY_LINE_COUNT
vertical_grid_capacity = int(PLOT_HISTORY / GRID_STEP) + 1
horizontal_grid_capacity = int(2 * PLOT_Y_LIMIT / GRID_STEP) + 1
for line_index in range(
    GRID_FIRST_LINE,
    GRID_FIRST_LINE + vertical_grid_capacity + horizontal_grid_capacity,
):
    telemetry_figure.linergb[line_index] = (0.18, 0.18, 0.18)


def periodic_kick(simulation_time: float) -> float:
    """Return a short torque pulse whose direction alternates each period."""
    cycle = int(simulation_time // KICK_PERIOD)
    time_in_cycle = simulation_time % KICK_PERIOD

    if time_in_cycle >= KICK_DURATION:
        return 0.0

    direction = 1.0 if cycle % 2 == 0 else -1.0
    return direction * KICK_TORQUE


def update_telemetry_figure(
    simulation_time: float,
    torque: float,
    angle: float,
    angular_velocity: float,
) -> None:
    """Append one telemetry sample and rebuild all three plot lines."""
    telemetry_samples.append(
        (simulation_time, torque, angle, angular_velocity)
    )
    point_count = len(telemetry_samples)
    relative_times = [
        sample[0] - simulation_time for sample in telemetry_samples
    ]

    for line_index in range(3):
        telemetry_figure.linepnt[line_index] = point_count
        telemetry_figure.linedata[line_index, : 2 * point_count : 2] = (
            relative_times
        )
        telemetry_figure.linedata[line_index, 1 : 2 * point_count : 2] = [
            sample[line_index + 1] for sample in telemetry_samples
        ]

    update_moving_grid(simulation_time)


def update_moving_grid(simulation_time: float) -> None:
    """Draw a one-unit grid whose time lines move with simulation time."""
    first_line = GRID_FIRST_LINE
    last_line = first_line + vertical_grid_capacity + horizontal_grid_capacity
    telemetry_figure.linepnt[first_line:last_line] = 0

    first_grid_time = math.ceil(
        (simulation_time - PLOT_HISTORY) / GRID_STEP
    ) * GRID_STEP
    grid_time = first_grid_time
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


def reset_telemetry() -> None:
    """Reset Python-owned plot state after MuJoCo resets MjData."""
    telemetry_samples.clear()
    telemetry_figure.linepnt[:] = 0


def main() -> None:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        next_log_time = 0.0
        next_plot_time = 0.0
        last_simulation_time = data.time
        viewport_logged = False

        while viewer.is_running():
            step_started = time.perf_counter()

            if data.time < last_simulation_time:
                reset_telemetry()
                next_log_time = data.time
                next_plot_time = data.time
                print("simulation reset detected; telemetry cleared")

            data.ctrl[motor_id] = periodic_kick(data.time)
            mujoco.mj_step(model, data)

            angle = data.joint("pivot").qpos[0]
            angular_velocity = data.joint("pivot").qvel[0]
            applied_torque = data.qfrc_actuator[pivot_dof_id]

            if data.time >= next_log_time:
                print(
                    f"t={data.time:5.2f} s  "
                    f"torque={data.ctrl[motor_id]:+4.1f} N*m  "
                    f"angle={angle:+6.3f} rad  "
                    f"velocity={angular_velocity:+6.3f} rad/s"
                )
                next_log_time += 0.1

            if data.time >= next_plot_time:
                update_telemetry_figure(
                    data.time,
                    applied_torque,
                    angle,
                    angular_velocity,
                )

                # viewer.viewport is the scene rectangle after the sidebars
                # have been removed. MjrRect uses window-relative bottom-left
                # pixel coordinates, so include the viewport's own offset.
                scene_viewport = viewer.viewport
                if not viewport_logged:
                    print(
                        "scene viewport: "
                        f"left={scene_viewport.left}, "
                        f"bottom={scene_viewport.bottom}, "
                        f"width={scene_viewport.width}, "
                        f"height={scene_viewport.height}"
                    )
                    viewport_logged = True

                plot_width = min(PLOT_WIDTH, scene_viewport.width)
                plot_height = min(PLOT_HEIGHT, scene_viewport.height)
                viewer.set_figures(
                    (
                        mujoco.MjrRect(
                            scene_viewport.left,
                            scene_viewport.bottom,
                            plot_width,
                            plot_height,
                        ),
                        telemetry_figure,
                    )
                )
                next_plot_time += PLOT_INTERVAL

            # Capture time before sync: the viewer can reset data during sync,
            # and the lower value will then be detected on the next iteration.
            last_simulation_time = data.time
            viewer.sync()

            # Keep wall-clock time close to simulation time without changing physics.
            sleep_time = model.opt.timestep - (time.perf_counter() - step_started)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
