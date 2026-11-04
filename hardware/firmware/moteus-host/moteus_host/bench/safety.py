"""Independent Tier-A motion limits and live stop checks."""

from __future__ import annotations

import math
from typing import Any

from .config import (
    CoastDownConfig,
    FrictionConfig,
    PositionStepConfig,
    SafetyConfig,
    TorqueSpeedConfig,
)


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
        "plateau_s", "plateau_tolerance_rev_s", "coast_stop_rev_s", "coast_settle_s", "coast_timeout_s",
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
    if config.coast_settle_s >= config.coast_timeout_s:
        raise ValueError("coast_down.coast_settle_s must be below coast_timeout_s")
    if config.direction not in (1, -1):
        raise ValueError("coast_down.direction must be 1 or -1")
    if not 50.0 <= config.sample_rate_hz <= 500.0:
        raise ValueError("coast_down.sample_rate_hz must be within 50..500 Hz")
    if config.response_timeout_s > 0.1:
        raise ValueError("coast_down.response_timeout_s must be <= 100 ms")


def validate_torque_speed(config: TorqueSpeedConfig, limits: SafetyConfig) -> None:
    """Tier B checks for the torque-speed envelope run (specs/torque-speed.md section 4)."""
    numeric = (
        "sample_rate_hz", "bus_tolerance_v", "saturation_alpha_rev_s2", "saturation_settle_s",
        "min_saturation_rev_s", "hold_s", "step_timeout_s", "cap_margin_rev_s", "sag_stop_v",
        "coast_stop_rev_s", "coast_settle_s", "coast_timeout_s", "max_velocity_rev_s",
        "max_torque_nm", "response_timeout_s", "kt_nm_per_a",
        "expected_noload_rpm_per_v",
    )
    for name in numeric:
        value = getattr(config, name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"torque_speed.{name} must be finite and positive")
    if not config.bus_voltages_v:
        raise ValueError("torque_speed.bus_voltages_v must list at least one voltage")
    for volts in config.bus_voltages_v:
        if not math.isfinite(volts) or volts <= 0:
            raise ValueError("torque_speed.bus_voltages_v entries must be finite and positive")
    if not config.torques_nm:
        raise ValueError("torque_speed.torques_nm must list at least one torque")
    if config.max_velocity_rev_s > limits.max_rotating_velocity_rev_s:
        raise ValueError("torque_speed.max_velocity_rev_s exceeds the Tier-B rotating velocity cap")
    if config.max_torque_nm > limits.max_rotating_torque_nm:
        raise ValueError("torque_speed.max_torque_nm exceeds the Tier-B rotating torque cap")
    for torque in config.torques_nm:
        if not math.isfinite(torque) or torque <= 0 or torque > config.max_torque_nm:
            raise ValueError("torque_speed.torques_nm entries must be positive and within max_torque_nm")
    if max(config.torques_nm) > 0.95 * config.max_torque_nm:
        raise ValueError("torque_speed.torques_nm must keep a 5 % margin under max_torque_nm "
                         f"(top step {max(config.torques_nm):g} Nm > {0.95 * config.max_torque_nm:g} Nm); "
                         "reported torque carries current noise")
    top_current_a = max(config.torques_nm) / config.kt_nm_per_a
    if top_current_a > limits.max_phase_current_a:
        raise ValueError(f"torque_speed.torques_nm top step needs {top_current_a:.2f} A by kt_nm_per_a, "
                         f"above safety.max_phase_current_a {limits.max_phase_current_a:g} A")
    if config.cap_margin_rev_s <= 2.0:
        raise ValueError("torque_speed.cap_margin_rev_s must exceed the firmware derate band (2.0 rev/s)")
    if config.min_saturation_rev_s >= config.max_velocity_rev_s - config.cap_margin_rev_s:
        raise ValueError("torque_speed.min_saturation_rev_s must stay below the velocity-capped threshold")
    # the sag stop must sit one 0.1 V telemetry step above the floor (1e-6 V float slack), so a
    # fold-back ends the step supply-limited before check_state trips on min_bus_voltage_v
    if min(config.bus_voltages_v) - config.sag_stop_v + 1e-6 < limits.min_bus_voltage_v + 0.1:
        raise ValueError(f"torque_speed.sag_stop_v {config.sag_stop_v:g} V at the lowest bus voltage "
                         f"{min(config.bus_voltages_v):g} V must stay one 0.1 V telemetry step above the floor "
                         f"(safety.min_bus_voltage_v {limits.min_bus_voltage_v:g} V)")
    if max(config.bus_voltages_v) > limits.max_bus_voltage_v:
        raise ValueError("torque_speed.bus_voltages_v exceeds the safety max_bus_voltage_v")
    if config.saturation_settle_s >= config.step_timeout_s:
        raise ValueError("torque_speed.saturation_settle_s must be below step_timeout_s")
    if config.coast_settle_s >= config.coast_timeout_s:
        raise ValueError("torque_speed.coast_settle_s must be below coast_timeout_s")
    if not config.directions or any(d not in (1, -1) for d in config.directions):
        raise ValueError("torque_speed.directions must be a subset of 1 and -1")
    if not 50.0 <= config.sample_rate_hz <= 500.0:
        raise ValueError("torque_speed.sample_rate_hz must be within 50..500 Hz")
    if config.response_timeout_s > 0.1:
        raise ValueError("torque_speed.response_timeout_s must be <= 100 ms")


FAULT_MODE = 1
LIMIT_CODE_MIN = 96
FAULT_NAMES = {
    32: "calibration fault",
    33: "motor driver fault",
    34: "over voltage",
    35: "encoder fault",
    36: "motor not configured",
    37: "pwm cycle overrun",
    38: "over temperature",
    39: "outside limit",
    40: "under voltage",
    41: "config changed",
    42: "theta invalid",
    43: "position invalid",
    44: "driver enable fault",
    45: "stop position deprecated",
    46: "timing violation",
    47: "bemf feedforward no accel",
    48: "invalid limits",
    49: "position control error",
    50: "velocity control error",
}


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
    mode = state.get("mode")
    if fault not in (None, 0) and (mode == FAULT_MODE or fault < LIMIT_CODE_MIN):
        name = FAULT_NAMES.get(fault, "unknown")
        raise SafetyStop(f"moteus fault {fault} ({name})")
    # Codes >= 96 in a control mode are limit indicators (96 max_velocity,
    # 97 max_power, 98 bus voltage, 99 max_current, ...), not faults; they stay
    # in the telemetry `fault` column and the voltage/current/torque checks
    # below still apply.

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
