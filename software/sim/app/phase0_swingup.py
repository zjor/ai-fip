"""Constrained swing-up simulation for the Phase 0 momentum gate.

State: pendulum angle from upright ``theta`` (rad, 0 = upright, pi = hanging),
pendulum rate, and absolute flywheel speed. The motor torque acts with opposite
signs on the wheel and on the pendulum body. Torque is limited by the moteus
voltage-circle envelope at the current wheel speed (``Actuator.available_torque_nm``)
and by the driver current limit, so torque fades as the wheel spins up.

Controller: energy pumping (bang-bang on the sign of the pendulum rate, scaled
by the energy error) until the energy is close to the upright value, then a
hand-over to the sampled LQR from ``app.phase0_control`` when the angle is
inside the catch cone. The wheel-speed weight in the LQR despins the wheel.

Reported per run: whether the catch succeeded, swing-up time, maximum wheel
speed as a fraction of the no-load speed, the required momentum
``H = I_w * max|omega_w|``, and torque statistics.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from app.phase0 import (
    RAD_S_PER_RPM,
    Actuator,
    DesignPoint,
    load_config,
    nominal_point,
    pessimistic_point,
)
from app.phase0_control import lqr_gain


@dataclass(frozen=True)
class SwingUpResult:
    caught: bool
    upright_settled: bool
    swing_up_time_s: float
    number_of_swings: int
    maximum_wheel_speed_rpm: float
    wheel_speed_fraction_of_no_load: float
    wheel_speed_at_catch_rpm: float
    required_momentum_nms: float
    maximum_torque_nm: float
    rms_torque_nm: float
    rms_current_a: float
    final_angle_deg: float
    final_wheel_speed_rpm: float
    reason: str


def simulate_swing_up(
    point: DesignPoint,
    actuator: Actuator,
    bus_voltage_v: float,
    *,
    energy_gain: float = 60.0,
    torque_limit_nm: float | None = None,
    catch_cone_deg: float = 30.0,
    energy_tolerance_fraction: float = 0.05,
    duration_s: float = 20.0,
    dt: float = 0.0005,
    sample_rate_hz: float = 500.0,
) -> SwingUpResult:
    gravity = point.gravity_coefficient_nm()
    inertia = point.pivot_inertia_kg_m2
    wheel_inertia = point.flywheel_inertia_kg_m2
    no_load_rad_s = actuator.no_load_rpm(bus_voltage_v) * RAD_S_PER_RPM
    gain = lqr_gain(point)

    def energy(theta: float, theta_rate: float) -> float:
        # Zero at upright rest; hanging at rest is -2G.
        return 0.5 * inertia * theta_rate**2 + gravity * (math.cos(theta) - 1.0)

    def derivative(current: np.ndarray, torque: float) -> np.ndarray:
        theta, theta_rate, _ = current
        return np.array(
            [
                theta_rate,
                (gravity * math.sin(theta) - torque) / inertia,
                torque / wheel_inertia,
            ]
        )

    def wrap(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    state = np.array([math.pi - 1e-3, 0.0, 0.0])  # hanging, tiny asymmetry
    steps = math.ceil(duration_s / dt)
    sample_steps = max(1, round(1.0 / sample_rate_hz / dt))
    torque = 0.0
    mode = "pump"
    catch_step: int | None = None
    swings = 0
    previous_rate_sign = 0.0
    maximum_wheel_speed = 0.0
    wheel_speed_at_catch = 0.0
    maximum_torque = 0.0
    torques: list[float] = []
    currents: list[float] = []
    settled_angles: list[float] = []
    settled_wheel: list[float] = []

    for step in range(steps):
        theta = wrap(float(state[0]))
        theta_rate = float(state[1])
        wheel_speed = float(state[2])

        if step % sample_steps == 0:
            if mode == "pump":
                energy_error = -energy(theta, theta_rate)  # positive while below upright
                if abs(theta) < math.radians(catch_cone_deg) and abs(energy_error) < (
                    energy_tolerance_fraction * 2.0 * gravity
                ):
                    mode = "catch"
                    catch_step = step
                    wheel_speed_at_catch = abs(wheel_speed)
                    request = float(-gain @ np.array([theta, theta_rate, wheel_speed]))
                else:
                    # Motor torque on the pendulum is -torque; pumping needs
                    # -torque * theta_rate > 0 when energy is short.
                    request = -energy_gain * energy_error * theta_rate
            else:
                request = float(-gain @ np.array([theta, theta_rate, wheel_speed]))
            limit = actuator.available_torque_nm(bus_voltage_v, wheel_speed)
            # Torque that would accelerate the wheel further is limited by the
            # envelope; braking torque only needs the current limit.
            braking = (request * wheel_speed) < 0.0
            if braking:
                limit = actuator.peak_torque_nm()
            if torque_limit_nm is not None and mode == "pump":
                limit = min(limit, torque_limit_nm)
            torque = float(np.clip(request, -limit, limit))

        k1 = derivative(state, torque)
        k2 = derivative(state + 0.5 * dt * k1, torque)
        k3 = derivative(state + 0.5 * dt * k2, torque)
        k4 = derivative(state + dt * k3, torque)
        state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        rate_sign = math.copysign(1.0, state[1]) if abs(state[1]) > 1e-6 else previous_rate_sign
        if mode == "pump" and previous_rate_sign and rate_sign != previous_rate_sign:
            swings += 1
        previous_rate_sign = rate_sign

        maximum_wheel_speed = max(maximum_wheel_speed, abs(float(state[2])))
        maximum_torque = max(maximum_torque, abs(torque))
        torques.append(torque)
        currents.append(abs(torque) / actuator.motor_kt_nm_per_a)
        if step * dt >= duration_s - 1.0:
            settled_angles.append(abs(wrap(float(state[0]))))
            settled_wheel.append(abs(float(state[2])))

        if mode == "catch" and abs(wrap(float(state[0]))) > math.radians(90.0):
            return _result(
                False, False, catch_step, dt, swings, maximum_wheel_speed, no_load_rad_s,
                wheel_speed_at_catch, wheel_inertia, maximum_torque, torques, currents,
                state, "fell after catch hand-over",
            )

    if mode != "catch":
        return _result(
            False, False, None, dt, swings, maximum_wheel_speed, no_load_rad_s,
            wheel_speed_at_catch, wheel_inertia, maximum_torque, torques, currents,
            state, "never reached the catch cone with enough energy",
        )
    settled = (
        math.degrees(max(settled_angles)) <= 2.0 and max(settled_wheel) / RAD_S_PER_RPM <= 100.0
    )
    reason = "caught and settled" if settled else (
        f"caught but not settled: {math.degrees(max(settled_angles)):.1f} deg, "
        f"{max(settled_wheel) / RAD_S_PER_RPM:.0f} rpm in the last second"
    )
    return _result(
        True, settled, catch_step, dt, swings, maximum_wheel_speed, no_load_rad_s,
        wheel_speed_at_catch, wheel_inertia, maximum_torque, torques, currents, state, reason,
    )


def _result(
    caught: bool,
    settled: bool,
    catch_step: int | None,
    dt: float,
    swings: int,
    maximum_wheel_speed: float,
    no_load_rad_s: float,
    wheel_speed_at_catch: float,
    wheel_inertia: float,
    maximum_torque: float,
    torques: list[float],
    currents: list[float],
    state: np.ndarray,
    reason: str,
) -> SwingUpResult:
    return SwingUpResult(
        caught=caught,
        upright_settled=settled,
        swing_up_time_s=(catch_step * dt) if catch_step is not None else math.inf,
        number_of_swings=swings,
        maximum_wheel_speed_rpm=maximum_wheel_speed / RAD_S_PER_RPM,
        wheel_speed_fraction_of_no_load=maximum_wheel_speed / no_load_rad_s,
        wheel_speed_at_catch_rpm=wheel_speed_at_catch / RAD_S_PER_RPM,
        required_momentum_nms=wheel_inertia * maximum_wheel_speed,
        maximum_torque_nm=maximum_torque,
        rms_torque_nm=math.sqrt(float(np.mean(np.square(torques)))) if torques else 0.0,
        rms_current_a=math.sqrt(float(np.mean(np.square(currents)))) if currents else 0.0,
        final_angle_deg=math.degrees(((float(state[0]) + math.pi) % (2 * math.pi)) - math.pi),
        final_wheel_speed_rpm=float(state[2]) / RAD_S_PER_RPM,
        reason=reason,
    )


def sweep(
    config, *, bus_voltages: tuple[float, ...], torque_limit_nm: float | None = None
) -> list[dict]:
    rows = []
    design = config.design
    points = {
        "nominal": nominal_point(design),
        "pessimistic": pessimistic_point(design),
    }
    for label, base in points.items():
        for wheel_inertia in (
            design.flywheel_inertia_kg_m2.low,
            design.flywheel_inertia_kg_m2.nominal,
            design.flywheel_inertia_kg_m2.high,
        ):
            point = DesignPoint(
                base.total_mass_kg, base.center_of_mass_m, base.pivot_inertia_kg_m2, wheel_inertia
            )
            for actuator in config.actuators:
                for bus in bus_voltages:
                    result = simulate_swing_up(point, actuator, bus, torque_limit_nm=torque_limit_nm)
                    rows.append(
                        {
                            "design": label,
                            "flywheel_inertia_kg_m2": wheel_inertia,
                            "actuator": actuator.driver.name,
                            "bus_voltage_v": bus,
                            "result": result,
                        }
                    )
    return rows


def render_sweep(rows: list[dict], maximum_fraction: float) -> str:
    lines = [
        "| Design | I_w | Driver | Bus | Caught | Settled | Time | Swings | Max wheel rpm (frac of no-load) | H required | RMS current | Gate |",
        "|---|---:|---|---:|:---:|:---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        r: SwingUpResult = row["result"]
        gate = "PASS" if (r.caught and r.upright_settled and r.wheel_speed_fraction_of_no_load <= maximum_fraction) else "FAIL"
        lines.append(
            f"| {row['design']} | {row['flywheel_inertia_kg_m2']:.4f} | {row['actuator']} | {row['bus_voltage_v']:.1f} V | "
            f"{'yes' if r.caught else 'no'} | {'yes' if r.upright_settled else 'no'} | "
            f"{r.swing_up_time_s:.1f} s | {r.number_of_swings} | {r.maximum_wheel_speed_rpm:.0f} ({r.wheel_speed_fraction_of_no_load:.2f}) | "
            f"{r.required_momentum_nms:.3f} Nms | {r.rms_current_a:.1f} A | {gate} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pessimistic", action="store_true")
    parser.add_argument("--actuator", choices=("c1", "r4.11"), default="r4.11")
    parser.add_argument("--bus-voltage", type=float, default=None)
    parser.add_argument("--flywheel-inertia", type=float, default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument(
        "--torque-limit",
        type=float,
        default=None,
        help="software torque cap during pumping, Nm (strategy study; default: hardware peak)",
    )
    args = parser.parse_args()

    config = load_config()
    if args.sweep:
        rows = sweep(
            config,
            bus_voltages=(config.minimum_bus_voltage_v, config.nominal_bus_voltage_v),
            torque_limit_nm=args.torque_limit,
        )
        print(render_sweep(rows, config.requirements.maximum_wheel_speed_fraction_at_catch))
        return

    point = pessimistic_point(config.design) if args.pessimistic else nominal_point(config.design)
    if args.flywheel_inertia is not None:
        point = DesignPoint(
            point.total_mass_kg, point.center_of_mass_m, point.pivot_inertia_kg_m2, args.flywheel_inertia
        )
    actuator = config.actuators[0 if args.actuator == "c1" else 1]
    bus = args.bus_voltage or config.minimum_bus_voltage_v
    result = simulate_swing_up(point, actuator, bus, torque_limit_nm=args.torque_limit)
    print(f"actuator: {actuator.name}, bus {bus:.1f} V, no-load {actuator.no_load_rpm(bus):.0f} rpm")
    print(f"design: {point}")
    for key, value in result.__dict__.items():
        print(f"  {key}: {value:.4g}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
