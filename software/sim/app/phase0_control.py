"""Provisional sampled-LQR check for Phase 0 sensor and latency budgeting."""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_are

from app.phase0 import Actuator, DesignPoint, actuator_result, load_config, nominal_point
from app.phase0 import pessimistic_point as _pessimistic_point


@dataclass(frozen=True)
class SensorScenario:
    sample_rate_hz: float = 500.0
    command_delay_ms: float = 5.0
    angle_noise_std_deg: float = 0.1
    angular_rate_noise_std_deg_s: float = 1.0
    wheel_speed_noise_std_rpm: float = 5.0


@dataclass(frozen=True)
class SimulationResult:
    passed: bool
    final_angle_deg: float
    maximum_angle_deg: float
    final_wheel_speed_rpm: float
    maximum_wheel_speed_rpm: float
    rms_torque_nm: float
    reason: str


def lqr_gain(point: DesignPoint) -> np.ndarray:
    """Return a continuous LQR gain for [angle, angle rate, wheel speed]."""
    gravity = point.gravity_coefficient_nm()
    a = np.array(
        [
            [0.0, 1.0, 0.0],
            [gravity / point.pivot_inertia_kg_m2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    b = np.array(
        [
            [0.0],
            [-1.0 / point.pivot_inertia_kg_m2],
            [1.0 / point.flywheel_inertia_kg_m2],
        ]
    )
    q = np.diag([120.0, 2.0, 0.015])
    r = np.array([[0.8]])
    solution = solve_continuous_are(a, b, q, r)
    return np.asarray(np.linalg.solve(r, b.T @ solution)).reshape(3)


def available_peak_torque_nm(
    actuator: Actuator, bus_voltage_v: float, wheel_speed_rad_s: float
) -> float:
    """Torque available at this wheel speed: moteus voltage circle + current limit."""
    return actuator.available_torque_nm(bus_voltage_v, wheel_speed_rad_s)


def simulate(
    point: DesignPoint,
    actuator: Actuator,
    bus_voltage_v: float,
    sensor: SensorScenario,
    *,
    initial_angle_deg: float = 10.0,
    duration_s: float = 5.0,
    seed: int = 0,
) -> SimulationResult:
    sample_period_s = 1.0 / sensor.sample_rate_hz
    integration_period_s = min(0.0005, sample_period_s / 4.0)
    steps = math.ceil(duration_s / integration_period_s)
    sample_steps = max(1, round(sample_period_s / integration_period_s))
    delay_steps = max(0, round(sensor.command_delay_ms / 1000.0 / integration_period_s))
    command_queue: deque[tuple[int, float]] = deque()
    rng = np.random.default_rng(seed)
    gain = lqr_gain(point)
    state = np.array([math.radians(initial_angle_deg), 0.0, 0.0])
    delayed_torque_request = 0.0
    applied_torque = 0.0
    maximum_angle = abs(state[0])
    maximum_wheel_speed = 0.0
    torques: list[float] = []
    settled_angles: list[float] = []
    settled_wheel_speeds: list[float] = []

    def derivative(current: np.ndarray, torque: float) -> np.ndarray:
        theta, theta_rate, _wheel_speed = current
        return np.array(
            [
                theta_rate,
                (
                    point.gravity_coefficient_nm() * math.sin(theta) - torque
                )
                / point.pivot_inertia_kg_m2,
                torque / point.flywheel_inertia_kg_m2,
            ]
        )

    for step in range(steps):
        if step % sample_steps == 0:
            measured = state.copy()
            measured[0] += math.radians(
                rng.normal(0.0, sensor.angle_noise_std_deg)
            )
            measured[1] += math.radians(
                rng.normal(0.0, sensor.angular_rate_noise_std_deg_s)
            )
            measured[2] += rng.normal(
                0.0, sensor.wheel_speed_noise_std_rpm
            ) * 2.0 * math.pi / 60.0
            new_torque_request = float(-gain @ measured)
            command_queue.append((step + delay_steps, new_torque_request))

        while command_queue and command_queue[0][0] <= step:
            _, delayed_torque_request = command_queue.popleft()

        torque_limit = available_peak_torque_nm(
            actuator, bus_voltage_v, state[2]
        )
        applied_torque = float(
            np.clip(delayed_torque_request, -torque_limit, torque_limit)
        )

        dt = integration_period_s
        k1 = derivative(state, applied_torque)
        k2 = derivative(state + 0.5 * dt * k1, applied_torque)
        k3 = derivative(state + 0.5 * dt * k2, applied_torque)
        k4 = derivative(state + dt * k3, applied_torque)
        state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        maximum_angle = max(maximum_angle, abs(state[0]))
        maximum_wheel_speed = max(maximum_wheel_speed, abs(state[2]))
        torques.append(applied_torque)
        if step * integration_period_s >= duration_s - 1.0:
            settled_angles.append(abs(state[0]))
            settled_wheel_speeds.append(abs(state[2]))

        if abs(state[0]) > math.radians(90.0):
            return SimulationResult(
                False,
                math.degrees(state[0]),
                math.degrees(maximum_angle),
                state[2] * 60.0 / (2.0 * math.pi),
                maximum_wheel_speed * 60.0 / (2.0 * math.pi),
                math.sqrt(float(np.mean(np.square(torques)))),
                "pendulum fell beyond 90 degrees",
            )

    final_angle_deg = math.degrees(state[0])
    final_wheel_speed_rpm = state[2] * 60.0 / (2.0 * math.pi)
    settled_angle_max_deg = math.degrees(max(settled_angles))
    settled_wheel_speed_max_rpm = max(settled_wheel_speeds) * 60.0 / (
        2.0 * math.pi
    )
    passed = settled_angle_max_deg <= 2.0 and settled_wheel_speed_max_rpm <= 100.0
    reason = (
        "settled inside angle and wheel-speed bounds"
        if passed
        else (
            f"last-second bounds exceeded: {settled_angle_max_deg:.2f} deg, "
            f"{settled_wheel_speed_max_rpm:.1f} rpm"
        )
    )
    return SimulationResult(
        passed,
        final_angle_deg,
        math.degrees(maximum_angle),
        final_wheel_speed_rpm,
        maximum_wheel_speed * 60.0 / (2.0 * math.pi),
        math.sqrt(float(np.mean(np.square(torques)))),
        reason,
    )


def pessimistic_point() -> DesignPoint:
    return _pessimistic_point(load_config().design)


def sweep_sensor_budget(
    point: DesignPoint,
    actuator: Actuator,
    bus_voltage_v: float,
    *,
    trials: int = 5,
) -> dict[str, list[tuple[float, int]]]:
    """Sweep one sensor dimension at a time around the reference scenario."""

    def passes(scenario: SensorScenario) -> int:
        return sum(
            simulate(
                point,
                actuator,
                bus_voltage_v,
                scenario,
                seed=seed,
            ).passed
            for seed in range(trials)
        )

    rates = [50.0, 75.0, 100.0, 200.0, 500.0, 1000.0]
    delays = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0]
    angle_noises = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    return {
        "sample_rate_hz": [
            (rate, passes(SensorScenario(sample_rate_hz=rate))) for rate in rates
        ],
        "command_delay_ms": [
            (delay, passes(SensorScenario(command_delay_ms=delay)))
            for delay in delays
        ],
        # Rate noise is scaled at 10 deg/s per degree of angle noise. This is a
        # scenario family, not a claim that the two errors are physically linked.
        "angle_noise_deg_with_10x_rate_noise_deg_s": [
            (
                noise,
                passes(
                    SensorScenario(
                        angle_noise_std_deg=noise,
                        angular_rate_noise_std_deg_s=noise * 10.0,
                    )
                ),
            )
            for noise in angle_noises
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pessimistic", action="store_true")
    parser.add_argument("--actuator", choices=("c1", "r4.11"), default="r4.11")
    parser.add_argument("--sample-rate-hz", type=float, default=500.0)
    parser.add_argument("--delay-ms", type=float, default=5.0)
    parser.add_argument("--angle-noise-deg", type=float, default=0.1)
    parser.add_argument("--rate-noise-deg-s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    config = load_config()
    point = pessimistic_point() if args.pessimistic else nominal_point(config.design)
    actuator = config.actuators[0 if args.actuator == "c1" else 1]
    if args.sweep:
        results = sweep_sensor_budget(
            point,
            actuator,
            config.minimum_bus_voltage_v,
            trials=args.trials,
        )
        print(f"actuator: {actuator.name}")
        print(f"design: {'pessimistic' if args.pessimistic else 'nominal'}")
        print(f"trials per point: {args.trials}")
        for dimension, values in results.items():
            print(f"\n{dimension}")
            for value, passed in values:
                print(f"  {value:g}: {passed}/{args.trials} pass")
        return

    scenario = SensorScenario(
        sample_rate_hz=args.sample_rate_hz,
        command_delay_ms=args.delay_ms,
        angle_noise_std_deg=args.angle_noise_deg,
        angular_rate_noise_std_deg_s=args.rate_noise_deg_s,
    )
    result = simulate(
        point,
        actuator,
        config.minimum_bus_voltage_v,
        scenario,
        seed=args.seed,
    )
    actuator_summary = actuator_result(
        actuator, point, config.requirements, config.minimum_bus_voltage_v
    )
    print(f"actuator: {actuator.name}")
    print(f"design: {'pessimistic' if args.pessimistic else 'nominal'}")
    print(f"sensor scenario: {scenario}")
    print(f"peak torque: {actuator_summary['peak_torque_nm']:.3f} Nm")
    print(f"result: {'PASS' if result.passed else 'FAIL'} — {result.reason}")
    print(f"maximum angle: {result.maximum_angle_deg:.2f} deg")
    print(f"final angle: {result.final_angle_deg:.3f} deg")
    print(f"maximum wheel speed: {result.maximum_wheel_speed_rpm:.1f} rpm")
    print(f"final wheel speed: {result.final_wheel_speed_rpm:.1f} rpm")
    print(f"RMS torque: {result.rms_torque_nm:.3f} Nm")


if __name__ == "__main__":
    main()
