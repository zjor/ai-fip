"""Independent Tier-A motion limits and live stop checks."""

from __future__ import annotations

import math
from typing import Any

from .config import CoastDownConfig, FrictionConfig, PositionStepConfig, SafetyConfig


class SafetyStop(RuntimeError):
    """Raised when a configured bench limit requires motion to stop."""


def validate_position_step(config: PositionStepConfig, limits: SafetyConfig) -> None:
    if config.max_torque_nm > limits.max_torque_nm:
        raise ValueError("position-step torque exceeds the Tier-A hard limit")
    if config.velocity_limit_rev_s > limits.max_abs_velocity_rev_s:
        raise ValueError("position-step velocity exceeds the Tier-A hard limit")
    if config.step_rev > limits.max_position_excursion_rev:
        raise ValueError("position-step excursion exceeds the Tier-A hard limit")


def validate_friction(config: FrictionConfig, limits: SafetyConfig) -> None:
    if config.max_torque_nm > limits.max_torque_nm:
        raise ValueError("friction torque exceeds the Tier-A hard limit")
    if config.velocity_limit_rev_s > limits.max_friction_velocity_rev_s:
        raise ValueError("friction velocity exceeds the Tier-A hard limit")
    if config.travel_rev > limits.max_friction_excursion_rev:
        raise ValueError("friction travel exceeds the Tier-A hard limit")


def validate_coast_down(config: CoastDownConfig, limits: SafetyConfig) -> None:
    """Tier B checks: every value must sit inside the tracked rotating caps."""
    numeric = (
        "sample_rate_hz", "spin_up_torque_nm", "accel_limit_rev_s2", "spin_up_timeout_s",
        "plateau_s", "plateau_tolerance_rev_s", "coast_stop_rev_s", "coast_timeout_s",
        "accel_target_rev_s", "accel_timeout_s", "max_velocity_rev_s", "max_torque_nm",
        "response_timeout_s", "kt_nm_per_a",
    )
    for name in numeric:
        value = getattr(config, name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"coast_down.{name} must be finite and positive")
    if not config.ladder_rpm:
        raise ValueError("coast_down.ladder_rpm must list at least one speed")
    for rpm in config.ladder_rpm:
        if not math.isfinite(rpm) or rpm <= 0:
            raise ValueError("coast_down.ladder_rpm entries must be finite and positive")
        if rpm / 60.0 >= config.max_velocity_rev_s:
            raise ValueError(
                f"coast_down.ladder_rpm {rpm:g} rpm reaches the {config.max_velocity_rev_s:g} rev/s cap"
            )
    if config.max_velocity_rev_s > limits.max_rotating_velocity_rev_s:
        raise ValueError("coast_down.max_velocity_rev_s exceeds the Tier-B rotating velocity cap")
    if config.max_torque_nm > limits.max_rotating_torque_nm:
        raise ValueError("coast_down.max_torque_nm exceeds the Tier-B rotating torque cap")
    if config.spin_up_torque_nm > config.max_torque_nm:
        raise ValueError("coast_down.spin_up_torque_nm exceeds coast_down.max_torque_nm")
    for torque in config.accel_torques_nm:
        if not math.isfinite(torque) or torque <= 0 or torque > config.max_torque_nm:
            raise ValueError("coast_down.accel_torques_nm entries must be positive and within max_torque_nm")
    if config.accel_target_rev_s >= config.max_velocity_rev_s:
        raise ValueError("coast_down.accel_target_rev_s must stay below max_velocity_rev_s")
    if config.coast_stop_rev_s >= min(config.ladder_rpm) / 60.0:
        raise ValueError("coast_down.coast_stop_rev_s must be below the lowest ladder speed")
    if config.direction not in (1, -1):
        raise ValueError("coast_down.direction must be 1 or -1")
    if not 50.0 <= config.sample_rate_hz <= 500.0:
        raise ValueError("coast_down.sample_rate_hz must be within 50..500 Hz")
    if config.response_timeout_s > 0.1:
        raise ValueError("coast_down.response_timeout_s must be <= 100 ms")


def check_state(
    state: dict[str, Any],
    limits: SafetyConfig,
    *,
    origin_position_rev: float | None = None,
    max_position_excursion_rev: float | None = None,
    max_abs_velocity_rev_s: float | None = None,
    max_torque_nm: float | None = None,
) -> None:
    required_fields = (
        "mode",
        "fault",
        "position_rev",
        "velocity_rev_s",
        "torque_nm",
        "q_current_a",
        "d_current_a",
        "bus_voltage_v",
        "controller_temperature_c",
    )
    missing = [name for name in required_fields if state.get(name) is None]
    if missing:
        raise SafetyStop(
            "missing safety-critical telemetry: " + ", ".join(missing)
        )

    fault = state.get("fault")
    if fault not in (None, 0):
        raise SafetyStop(f"moteus fault {fault}")

    voltage = state.get("bus_voltage_v")
    if voltage is not None and not (
        limits.min_bus_voltage_v <= voltage <= limits.max_bus_voltage_v
    ):
        raise SafetyStop(f"bus voltage outside limits: {voltage:.3f} V")

    checks = (
        (
            "controller temperature",
            state.get("controller_temperature_c"),
            limits.max_controller_temperature_c,
            "°C",
        ),
        (
            "motor temperature",
            state.get("motor_temperature_c"),
            limits.max_motor_temperature_c,
            "°C",
        ),
        (
            "Q phase current",
            abs(state["q_current_a"]) if state.get("q_current_a") is not None else None,
            limits.max_phase_current_a,
            "A",
        ),
        (
            "D phase current",
            abs(state["d_current_a"]) if state.get("d_current_a") is not None else None,
            limits.max_phase_current_a,
            "A",
        ),
        (
            "torque",
            abs(state["torque_nm"]) if state.get("torque_nm") is not None else None,
            max_torque_nm or limits.max_torque_nm,
            "N·m",
        ),
        (
            "velocity",
            abs(state["velocity_rev_s"])
            if state.get("velocity_rev_s") is not None
            else None,
            max_abs_velocity_rev_s or limits.max_abs_velocity_rev_s,
            "rev/s",
        ),
    )
    for label, value, maximum, unit in checks:
        if value is not None and math.isfinite(value) and value > maximum:
            raise SafetyStop(f"{label} exceeds limit: {value:.3f} {unit}")

    position = state.get("position_rev")
    if (
        origin_position_rev is not None
        and position is not None
        and math.isfinite(position)
        and abs(position - origin_position_rev)
        > (max_position_excursion_rev or limits.max_position_excursion_rev)
    ):
        raise SafetyStop(
            "position excursion exceeds limit: "
            f"{abs(position - origin_position_rev):.4f} rev"
        )
