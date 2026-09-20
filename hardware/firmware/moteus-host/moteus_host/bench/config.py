"""Load and validate tracked bench defaults."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "bench.toml"


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
class BenchConfig:
    common: CommonConfig
    safety: SafetyConfig
    inspect: InspectConfig
    position_step: PositionStepConfig
    friction: FrictionConfig


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
