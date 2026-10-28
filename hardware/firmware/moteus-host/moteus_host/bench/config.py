"""Load and validate tracked bench defaults."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "bench.toml"
# moteus-host/bench.toml -> moteus-host -> firmware -> hardware -> repo root
REPO_ROOT = DEFAULT_CONFIG_PATH.parents[3]


@dataclass(frozen=True)
class CommonConfig:
    controller_id: int
    watchdog_timeout_s: float
    output_root: Path


@dataclass(frozen=True)
class SafetyConfig:
    min_bus_voltage_v: float
    max_bus_voltage_v: float
    max_controller_temperature_c: float
    max_motor_temperature_c: float
    max_phase_current_a: float
    max_abs_velocity_rev_s: float
    max_torque_nm: float
    max_position_excursion_rev: float
    max_friction_velocity_rev_s: float
    max_friction_excursion_rev: float
    # Tier B (retained rotating assembly and guard): docs/hardware/specs/coast-down.md
    max_rotating_velocity_rev_s: float = 55.0
    max_rotating_torque_nm: float = 0.20


@dataclass(frozen=True)
class InspectConfig:
    duration_s: float
    sample_rate_hz: float


@dataclass(frozen=True)
class PositionStepConfig:
    sample_rate_hz: float
    step_rev: float
    hold_s: float
    settle_s: float
    cycles: int
    max_torque_nm: float
    velocity_limit_rev_s: float
    accel_limit_rev_s2: float


@dataclass(frozen=True)
class FrictionConfig:
    sample_rate_hz: float
    travel_rev: float
    sweep_center_rev: float
    velocity_limit_rev_s: float
    accel_limit_rev_s2: float
    max_torque_nm: float
    settle_s: float
    turnaround_s: float
    move_timeout_s: float
    position_bound_margin_rev: float


@dataclass(frozen=True)
class ResponseTimingConfig:
    sample_rate_hz: float = 200.0
    pulse_torque_nm: float = 0.01
    pulse_s: float = 0.02
    repetitions_per_direction: int = 2
    quiet_s: float = 0.25
    quiet_timeout_s: float = 5.0
    quiet_velocity_rev_s: float = 0.01
    quiet_span_rev: float = 0.0005
    observe_s: float = 0.15
    response_timeout_s: float = 0.05
    max_velocity_rev_s: float = 0.25
    max_excursion_rev: float = 0.05


@dataclass(frozen=True)
class TorqueConstantConfig:
    sample_rate_hz: float = 200.0
    plateau_s: float = 3.0
    max_plateau_s: float = 10.0
    ramp_s: float = 0.25
    rest_s: float = 5.0
    baseline_s: float = 0.25
    max_current_a: float = 8.0
    current_overshoot_a: float = 0.5
    max_reported_torque_nm: float = 0.30
    session_i2t_budget_a2s: float = 2500.0
    max_velocity_rev_s: float = 0.05
    settle_max_velocity_rev_s: float = 0.15
    max_excursion_rev: float = 0.01
    response_timeout_s: float = 0.05
    max_rise_torque_nm: float = 0.05
    rise_step_s: float = 0.5
    rise_prehold_nm: float = 0.01
    rise_prehold_s: float = 0.25
    rise_sample_rate_hz: float = 400.0
    nominal_kt_nm_per_a: float = 0.025


@dataclass(frozen=True)
class CoastDownConfig:
    """Tier B coast-down and acceleration runs on the test disc (specs/coast-down.md)."""

    sample_rate_hz: float = 200.0
    ladder_rpm: tuple[float, ...] = (500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0)
    direction: int = 1
    spin_up_torque_nm: float = 0.10
    accel_limit_rev_s2: float = 20.0
    spin_up_timeout_s: float = 15.0
    plateau_s: float = 3.0
    plateau_tolerance_rev_s: float = 0.5
    coast_stop_rev_s: float = 0.2
    coast_settle_s: float = 0.25
    coast_timeout_s: float = 120.0
    accel_torques_nm: tuple[float, ...] = (0.05, 0.10, 0.15)
    accel_target_rev_s: float = 30.0
    accel_timeout_s: float = 10.0
    max_velocity_rev_s: float = 55.0
    max_torque_nm: float = 0.20
    response_timeout_s: float = 0.05
    inertia_record: str = "docs/hardware/measurements/2026-10-28-test-disc.json"
    kt_nm_per_a: float = 0.0244


@dataclass(frozen=True)
class BenchConfig:
    common: CommonConfig
    safety: SafetyConfig
    inspect: InspectConfig
    position_step: PositionStepConfig
    friction: FrictionConfig
    response_timing: ResponseTimingConfig = ResponseTimingConfig()
    torque_constant: TorqueConstantConfig = TorqueConstantConfig()
    coast_down: CoastDownConfig = CoastDownConfig()


def _coast_down(section: dict[str, Any]) -> CoastDownConfig:
    values = dict(section)
    for key in ("ladder_rpm", "accel_torques_nm"):
        if key in values:
            values[key] = tuple(float(item) for item in values[key])
    if "direction" in values:
        values["direction"] = int(values["direction"])
    return CoastDownConfig(**values)


def _positive(section: dict[str, Any], key: str) -> float:
    value = float(section[key])
    if value <= 0.0:
        raise ValueError(f"{key} must be positive")
    return value


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> BenchConfig:
    document = tomllib.loads(path.read_text())
    common = document["common"]
    safety = document["safety"]
    inspect = document["inspect"]
    position_step = document["position_step"]
    friction = document["friction"]

    config = BenchConfig(
        response_timing=ResponseTimingConfig(**document.get("response_timing", {})),
        torque_constant=TorqueConstantConfig(**document.get("torque_constant", {})),
        coast_down=_coast_down(document.get("coast_down", {})),
        common=CommonConfig(
            controller_id=int(common["controller_id"]),
            watchdog_timeout_s=_positive(common, "watchdog_timeout_s"),
            output_root=Path(common["output_root"]),
        ),
        safety=SafetyConfig(
            min_bus_voltage_v=_positive(safety, "min_bus_voltage_v"),
            max_bus_voltage_v=_positive(safety, "max_bus_voltage_v"),
            max_controller_temperature_c=_positive(
                safety, "max_controller_temperature_c"
            ),
            max_motor_temperature_c=_positive(
                safety, "max_motor_temperature_c"
            ),
            max_phase_current_a=_positive(safety, "max_phase_current_a"),
            max_abs_velocity_rev_s=_positive(
                safety, "max_abs_velocity_rev_s"
            ),
            max_torque_nm=_positive(safety, "max_torque_nm"),
            max_position_excursion_rev=_positive(
                safety, "max_position_excursion_rev"
            ),
            max_friction_velocity_rev_s=_positive(
                safety, "max_friction_velocity_rev_s"
            ),
            max_friction_excursion_rev=_positive(
                safety, "max_friction_excursion_rev"
            ),
            max_rotating_velocity_rev_s=float(safety.get("max_rotating_velocity_rev_s", 55.0)),
            max_rotating_torque_nm=float(safety.get("max_rotating_torque_nm", 0.20)),
        ),
        inspect=InspectConfig(
            duration_s=_positive(inspect, "duration_s"),
            sample_rate_hz=_positive(inspect, "sample_rate_hz"),
        ),
        position_step=PositionStepConfig(
            sample_rate_hz=_positive(position_step, "sample_rate_hz"),
            step_rev=_positive(position_step, "step_rev"),
            hold_s=_positive(position_step, "hold_s"),
            settle_s=_positive(position_step, "settle_s"),
            cycles=int(position_step["cycles"]),
            max_torque_nm=_positive(position_step, "max_torque_nm"),
            velocity_limit_rev_s=_positive(
                position_step, "velocity_limit_rev_s"
            ),
            accel_limit_rev_s2=_positive(
                position_step, "accel_limit_rev_s2"
            ),
        ),
        friction=FrictionConfig(
            sample_rate_hz=_positive(friction, "sample_rate_hz"),
            travel_rev=_positive(friction, "travel_rev"),
            sweep_center_rev=float(friction["sweep_center_rev"]),
            velocity_limit_rev_s=_positive(friction, "velocity_limit_rev_s"),
            accel_limit_rev_s2=_positive(friction, "accel_limit_rev_s2"),
            max_torque_nm=_positive(friction, "max_torque_nm"),
            settle_s=_positive(friction, "settle_s"),
            turnaround_s=_positive(friction, "turnaround_s"),
            move_timeout_s=_positive(friction, "move_timeout_s"),
            position_bound_margin_rev=_positive(
                friction, "position_bound_margin_rev"
            ),
        ),
    )
    if config.common.controller_id <= 0:
        raise ValueError("controller_id must be positive")
    if config.safety.max_bus_voltage_v <= config.safety.min_bus_voltage_v:
        raise ValueError("max_bus_voltage_v must exceed min_bus_voltage_v")
    if config.position_step.cycles <= 0:
        raise ValueError("cycles must be positive")
    if not math.isfinite(config.friction.sweep_center_rev):
        raise ValueError("sweep_center_rev must be finite")
    return config
