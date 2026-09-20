"""Independent Tier-A motion limits and live stop checks."""

from __future__ import annotations

import math
from typing import Any

from .config import FrictionConfig, PositionStepConfig, SafetyConfig


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


def check_state(
    state: dict[str, Any],
    limits: SafetyConfig,
    *,
    origin_position_rev: float | None = None,
    max_position_excursion_rev: float | None = None,
    max_abs_velocity_rev_s: float | None = None,
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
            limits.max_torque_nm,
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
