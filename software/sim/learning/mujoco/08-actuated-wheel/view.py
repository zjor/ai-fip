"""Swing up, balance, reject random kicks, and plot fixed-frame telemetry."""

import math
import time
from collections import deque
from importlib.util import find_spec
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from controller import SwingUpLQRController, reduced_state
from disturbance import RandomTangentialKicks


LOG_INTERVAL_S = 0.05
PLAYBACK_SPEED = 0.5
PLOT_HISTORY_S = 10.0
PLOT_INTERVAL_S = 0.02
PANEL_WIDTH = 576
PANEL_HEIGHT = 390
PANEL_GAP = 2
HEADER_HEIGHT = 54
HEADER_LEFT_PADDING = 14
HEADER_TOP_PADDING = 14
CHART_LEFT_GUTTER = 72
CHART_RIGHT_PADDING = 12
CHART_TOP_PADDING = 10
CHART_BOTTOM_GUTTER = 28
FIGURE_RGB = (20, 20, 20)
PANE_RGB = (6, 6, 6)
FRAME_RGB = (170, 170, 170)
TEXT_RGB = (235, 235, 235)

PLOT_DEFINITIONS = (
    ("u", (255, 166, 38), 0.1),
    ("ω  (rad/s)", (64, 191, 255), 1.0),
    ("τ  (N·m)", (76, 230, 102), 0.1),
    ("θ  (deg)", (255, 89, 76), 1.0),
)

scene_path = Path(__file__).with_name("scene.xml").resolve()
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)
actuator_id = model.actuator("wheel-torque").id
controller = SwingUpLQRController(model)
kicks = RandomTangentialKicks(seed=7)


def dejavu_font(size: int) -> ImageFont.FreeTypeFont:
    """Load the Unicode-capable font bundled with Matplotlib."""
    matplotlib_spec = find_spec("matplotlib")
    if matplotlib_spec is None or matplotlib_spec.origin is None:
        raise RuntimeError("Matplotlib installation is required for plot captions")
    font_path = (
        Path(matplotlib_spec.origin).parent
        / "mpl-data"
        / "fonts"
        / "ttf"
        / "DejaVuSans.ttf"
    )
    return ImageFont.truetype(font_path, size=size)


HEADER_FONT = dejavu_font(26)
AXIS_FONT = dejavu_font(14)

sample_capacity = int(PLOT_HISTORY_S / PLOT_INTERVAL_S) + 1
telemetry_samples: deque[tuple[float, float, float, float, float]] = deque(
    maxlen=sample_capacity
)


def padded_range(values: list[float], minimum_half_range: float) -> tuple[float, float]:
    """Return a data range with 20% of its span added at each edge."""
    lower = min(values)
    upper = max(values)
    center = 0.5 * (lower + upper)
    half_range = max(0.7 * (upper - lower), minimum_half_range)
    return center - half_range, center + half_range


def update_telemetry(
    simulation_time: float,
    requested_control: float,
    wheel_speed: float,
    motor_torque: float,
    rod_angle_deg: float,
) -> None:
    telemetry_samples.append(
        (
            simulation_time,
            requested_control,
            wheel_speed,
            motor_torque,
            rod_angle_deg,
        )
    )


def chart_frame(width: int, height: int) -> tuple[int, int, int, int]:
    """Return fixed pixel bounds for the line-drawing area."""
    left = min(CHART_LEFT_GUTTER, max(1, width // 3))
    right = max(left + 1, width - CHART_RIGHT_PADDING - 1)
    top = min(CHART_TOP_PADDING, max(0, height // 4))
    bottom = max(top + 1, height - CHART_BOTTOM_GUTTER - 1)
    return left, top, right, bottom


def format_scale_value(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1000.0 or (0.0 < magnitude < 0.01):
        return f"{value:.1e}"
    return f"{value:.2f}"


def render_header(title: str, width: int, height: int) -> np.ndarray:
    """Render a standalone, left-aligned panel header."""
    image = Image.new("RGB", (width, height), FIGURE_RGB)
    draw = ImageDraw.Draw(image)
    bounds = draw.textbbox((0, 0), title, font=HEADER_FONT)
    draw.text(
        (HEADER_LEFT_PADDING, HEADER_TOP_PADDING - bounds[1]),
        title,
        font=HEADER_FONT,
        fill=TEXT_RGB,
    )
    return np.asarray(image)


def render_chart(signal_index: int, width: int, height: int) -> np.ndarray:
    """Render one signal into a fixed pixel-space chart body."""
    image = Image.new("RGB", (width, height), FIGURE_RGB)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = chart_frame(width, height)
    draw.rectangle((left, top, right, bottom), fill=PANE_RGB, outline=FRAME_RGB, width=2)

    samples = list(telemetry_samples)
    values = [sample[signal_index] for sample in samples] or [0.0]
    y_min, y_max = padded_range(
        values, PLOT_DEFINITIONS[signal_index - 1][2]
    )

    draw.text(
        (left - 8, top),
        format_scale_value(y_max),
        font=AXIS_FONT,
        fill=TEXT_RGB,
        anchor="ra",
    )
    draw.text(
        (left - 8, bottom),
        format_scale_value(y_min),
        font=AXIS_FONT,
        fill=TEXT_RGB,
        anchor="rd",
    )
    draw.text((left, bottom + 5), "-10", font=AXIS_FONT, fill=TEXT_RGB, anchor="la")
    draw.text((right, bottom + 5), "0", font=AXIS_FONT, fill=TEXT_RGB, anchor="ra")

    if samples:
        newest_time = samples[-1][0]
        x_span = right - left
        y_span = bottom - top
        value_span = y_max - y_min
        points = [
            (
                round(left + ((sample[0] - newest_time + PLOT_HISTORY_S) / PLOT_HISTORY_S) * x_span),
                round(bottom - ((sample[signal_index] - y_min) / value_span) * y_span),
            )
            for sample in samples
        ]
        color = PLOT_DEFINITIONS[signal_index - 1][1]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        else:
            draw.line(points, fill=color, width=3, joint="curve")

    return np.asarray(image)


def panel_rectangles(
    viewport: mujoco.MjrRect,
) -> list[tuple[mujoco.MjrRect, mujoco.MjrRect]]:
    """Return fixed header/body rectangles, stacked at the viewport's top-right."""
    panel_count = len(PLOT_DEFINITIONS)
    gap = PANEL_GAP if viewport.height > panel_count * PANEL_GAP else 0
    available_height = viewport.height - gap * (panel_count - 1)
    panel_height = min(PANEL_HEIGHT, max(2, available_height // panel_count))
    header_height = min(HEADER_HEIGHT, max(1, panel_height // 3))
    body_height = panel_height - header_height
    panel_width = min(PANEL_WIDTH, viewport.width)
    panel_left = viewport.left + viewport.width - panel_width
    viewport_top = viewport.bottom + viewport.height

    rectangles = []
    for index in range(panel_count):
        panel_top = viewport_top - index * (panel_height + gap)
        header_bottom = panel_top - header_height
        body_bottom = panel_top - panel_height
        rectangles.append(
            (
                mujoco.MjrRect(
                    panel_left,
                    header_bottom,
                    panel_width,
                    header_height,
                ),
                mujoco.MjrRect(
                    panel_left,
                    body_bottom,
                    panel_width,
                    body_height,
                ),
            )
        )
    return rectangles


def telemetry_overlays(
    viewport: mujoco.MjrRect,
) -> list[tuple[mujoco.MjrRect, np.ndarray]]:
    """Build separate header and fixed-frame chart images for every signal."""
    overlays = []
    for signal_index, (header_rect, body_rect) in enumerate(
        panel_rectangles(viewport), start=1
    ):
        title = PLOT_DEFINITIONS[signal_index - 1][0]
        overlays.append(
            (
                header_rect,
                render_header(title, header_rect.width, header_rect.height),
            )
        )
        overlays.append(
            (
                body_rect,
                render_chart(signal_index, body_rect.width, body_rect.height),
            )
        )
    return overlays


def reset_telemetry() -> None:
    telemetry_samples.clear()


def initialize() -> None:
    """Reset close to the hanging equilibrium with a tiny asymmetry."""
    mujoco.mj_resetData(model, data)
    data.joint("pivot").qpos[0] = math.pi - 1e-3
    mujoco.mj_forward(model, data)
    controller.reset()
    kicks.reset()
    reset_telemetry()


def main() -> None:
    initialize()
    print(f"LQR gain [angle, angle rate, absolute wheel rate]: {controller.gain}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        next_log_time = 0.0
        next_plot_time = 0.0
        last_simulation_time = data.time
        previous_mode = controller.mode

        while viewer.is_running():
            step_started = time.perf_counter()

            if data.time < last_simulation_time:
                initialize()
                next_log_time = 0.0
                next_plot_time = 0.0
                previous_mode = controller.mode
                print("simulation reset near the hanging position")

            command = controller.update(data)
            data.ctrl[actuator_id] = command.applied_torque_nm
            kick = kicks.apply(model, data, enabled=command.mode == "lqr")

            if command.mode != previous_mode:
                print(f"t={data.time:.3f} s  mode -> {command.mode}")
                previous_mode = command.mode
            if kick.started:
                print(
                    f"t={data.time:.3f} s  KICK #{kick.number}: "
                    f"{kick.force_n:+.3f} N perpendicular to rod"
                )

            mujoco.mj_step(model, data)

            angle, pivot_rate, absolute_wheel_rate = reduced_state(data)
            motor_torque = float(data.actuator_force[actuator_id])

            if data.time >= next_log_time:
                print(
                    f"t={data.time:5.2f} s  mode={command.mode:8s}  "
                    f"pendulum={math.degrees(angle):+7.2f} deg  "
                    f"rate={pivot_rate:+8.3f} rad/s  "
                    f"wheel={absolute_wheel_rate:+9.3f} rad/s  "
                    f"torque={command.applied_torque_nm:+.3f} N*m  "
                    f"kick={kick.force_n:+.2f} N"
                )
                next_log_time += LOG_INTERVAL_S

            if data.time >= next_plot_time:
                update_telemetry(
                    data.time,
                    command.requested_torque_nm,
                    absolute_wheel_rate,
                    motor_torque,
                    math.degrees(angle),
                )
                viewer.set_images(telemetry_overlays(viewer.viewport))
                next_plot_time += PLOT_INTERVAL_S

            last_simulation_time = data.time
            viewer.sync()

            wall_step_duration = model.opt.timestep / PLAYBACK_SPEED
            sleep_time = wall_step_duration - (time.perf_counter() - step_started)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
