"""Analytical pre-check for the Phase 0 hardware feasibility gate.

This module intentionally keeps unknowns visible. In particular, a driver
continuous-current rating is not treated as proof of the motor's thermal
continuous torque, and flywheel momentum is not marked as sufficient until an
honest swing-up simulation provides the required momentum.
"""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any


GRAVITY_M_S2 = 9.81
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "phase0.toml"


@dataclass(frozen=True)
class Range3:
    low: float
    nominal: float
    high: float

    @classmethod
    def from_list(cls, values: list[float], name: str) -> "Range3":
        if len(values) != 3:
            raise ValueError(f"{name} must contain [low, nominal, high]")
        low, nominal, high = (float(value) for value in values)
        if not low <= nominal <= high:
            raise ValueError(f"{name} must be ordered low <= nominal <= high")
        if low <= 0:
            raise ValueError(f"{name} values must be positive")
        return cls(low, nominal, high)


@dataclass(frozen=True)
class Design:
    total_mass_kg: Range3
    center_of_mass_m: Range3
    pivot_inertia_kg_m2: Range3
    flywheel_inertia_kg_m2: Range3


@dataclass(frozen=True)
class Requirements:
    target_recovery_angle_deg: float
    minimum_peak_margin: float
    preferred_peak_margin: float
    minimum_continuous_torque_nm: float
    maximum_delay_fraction_of_time_constant: float
    maximum_sample_period_fraction_of_time_constant: float


@dataclass(frozen=True)
class Actuator:
    name: str
    motor_kv_rpm_per_v: float
    motor_kt_nm_per_a: float
    driver_peak_phase_current_a: float
    driver_continuous_phase_current_a: float
    motor_peak_torque_nm: float
    motor_max_rpm: float

    def peak_torque_nm(self) -> float:
        return min(
            self.motor_kt_nm_per_a * self.driver_peak_phase_current_a,
            self.motor_peak_torque_nm,
        )

    def driver_continuous_torque_proxy_nm(self) -> float:
        return min(
            self.motor_kt_nm_per_a * self.driver_continuous_phase_current_a,
            self.motor_peak_torque_nm,
        )

    def no_load_rpm(self, bus_voltage_v: float) -> float:
        return min(self.motor_kv_rpm_per_v * bus_voltage_v, self.motor_max_rpm)


@dataclass(frozen=True)
class Config:
    design: Design
    requirements: Requirements
    minimum_bus_voltage_v: float
    actuators: tuple[Actuator, ...]


@dataclass(frozen=True)
class DesignPoint:
    total_mass_kg: float
    center_of_mass_m: float
    pivot_inertia_kg_m2: float
    flywheel_inertia_kg_m2: float

    def gravity_coefficient_nm(self) -> float:
        return self.total_mass_kg * GRAVITY_M_S2 * self.center_of_mass_m

    def gravity_torque_nm(self, angle_deg: float) -> float:
        return self.gravity_coefficient_nm() * math.sin(math.radians(angle_deg))

    def upright_growth_rate_per_s(self) -> float:
        return math.sqrt(self.gravity_coefficient_nm() / self.pivot_inertia_kg_m2)

    def upright_time_constant_s(self) -> float:
        return 1.0 / self.upright_growth_rate_per_s()

    def swing_up_energy_j(self) -> float:
        return 2.0 * self.gravity_coefficient_nm()


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    design_raw = raw["design"]
    design = Design(
        total_mass_kg=Range3.from_list(design_raw["total_mass_kg"], "total_mass_kg"),
        center_of_mass_m=Range3.from_list(
            design_raw["center_of_mass_m"], "center_of_mass_m"
        ),
        pivot_inertia_kg_m2=Range3.from_list(
            design_raw["pivot_inertia_kg_m2"], "pivot_inertia_kg_m2"
        ),
        flywheel_inertia_kg_m2=Range3.from_list(
            design_raw["flywheel_inertia_kg_m2"], "flywheel_inertia_kg_m2"
        ),
    )
    requirements = Requirements(**raw["requirements"])
    actuators = tuple(Actuator(**item) for item in raw["actuators"])
    return Config(
        design=design,
        requirements=requirements,
        minimum_bus_voltage_v=float(raw["power"]["minimum_bus_voltage_v"]),
        actuators=actuators,
    )


def nominal_point(design: Design) -> DesignPoint:
    return DesignPoint(
        design.total_mass_kg.nominal,
        design.center_of_mass_m.nominal,
        design.pivot_inertia_kg_m2.nominal,
        design.flywheel_inertia_kg_m2.nominal,
    )


def design_corners(design: Design) -> list[DesignPoint]:
    values = (
        (design.total_mass_kg.low, design.total_mass_kg.high),
        (design.center_of_mass_m.low, design.center_of_mass_m.high),
        (design.pivot_inertia_kg_m2.low, design.pivot_inertia_kg_m2.high),
        (design.flywheel_inertia_kg_m2.low, design.flywheel_inertia_kg_m2.high),
    )
    return [DesignPoint(*point) for point in product(*values)]


def recoverable_angle_deg(
    available_torque_nm: float, gravity_coefficient_nm: float, safety_margin: float
) -> float:
    ratio = available_torque_nm / (safety_margin * gravity_coefficient_nm)
    if ratio >= 1.0:
        return 90.0
    return math.degrees(math.asin(max(0.0, ratio)))


def angular_momentum_nms(flywheel_inertia_kg_m2: float, speed_rpm: float) -> float:
    speed_rad_s = speed_rpm * 2.0 * math.pi / 60.0
    return flywheel_inertia_kg_m2 * speed_rad_s


def actuator_result(
    actuator: Actuator,
    point: DesignPoint,
    requirements: Requirements,
    bus_voltage_v: float,
) -> dict[str, Any]:
    peak_torque = actuator.peak_torque_nm()
    gravity_torque = point.gravity_torque_nm(
        requirements.target_recovery_angle_deg
    )
    no_load_rpm = actuator.no_load_rpm(bus_voltage_v)
    return {
        "name": actuator.name,
        "peak_torque_nm": peak_torque,
        "driver_continuous_torque_proxy_nm": actuator.driver_continuous_torque_proxy_nm(),
        "no_load_rpm_at_minimum_bus_voltage": no_load_rpm,
        "peak_margin_at_target_angle": peak_torque / gravity_torque,
        "recoverable_angle_deg_at_minimum_margin": recoverable_angle_deg(
            peak_torque,
            point.gravity_coefficient_nm(),
            requirements.minimum_peak_margin,
        ),
        "flywheel_momentum_capacity_nms": angular_momentum_nms(
            point.flywheel_inertia_kg_m2, no_load_rpm
        ),
    }


def build_report(config: Config) -> dict[str, Any]:
    nominal = nominal_point(config.design)
    corners = design_corners(config.design)
    worst_gravity = max(corners, key=lambda point: point.gravity_coefficient_nm())
    fastest_instability = min(corners, key=lambda point: point.upright_time_constant_s())
    minimum_flywheel = min(corners, key=lambda point: point.flywheel_inertia_kg_m2)

    nominal_actuators = [
        actuator_result(
            actuator,
            nominal,
            config.requirements,
            config.minimum_bus_voltage_v,
        )
        for actuator in config.actuators
    ]
    robust_actuators = []
    for actuator in config.actuators:
        result = actuator_result(
            actuator,
            worst_gravity,
            config.requirements,
            config.minimum_bus_voltage_v,
        )
        result["peak_gate"] = (
            "PASS"
            if result["peak_margin_at_target_angle"]
            >= config.requirements.minimum_peak_margin
            else "FAIL"
        )
        result["driver_continuous_proxy_gate"] = (
            "PASS"
            if result["driver_continuous_torque_proxy_nm"]
            >= config.requirements.minimum_continuous_torque_nm
            else "FAIL"
        )
        result["motor_thermal_gate"] = "UNKNOWN"
        result["swing_up_momentum_gate"] = "UNKNOWN"
        robust_actuators.append(result)

    fastest_time_constant = fastest_instability.upright_time_constant_s()
    return {
        "status": "NOT READY",
        "reason": (
            "motor thermal torque, cogging, measured mass distribution, validated "
            "sensor/estimator performance, and required swing-up momentum are not yet established"
        ),
        "nominal_design": {
            **asdict(nominal),
            "gravity_coefficient_nm": nominal.gravity_coefficient_nm(),
            "gravity_torque_at_target_nm": nominal.gravity_torque_nm(
                config.requirements.target_recovery_angle_deg
            ),
            "swing_up_energy_j": nominal.swing_up_energy_j(),
            "upright_time_constant_ms": nominal.upright_time_constant_s() * 1000.0,
        },
        "worst_case": {
            "gravity_coefficient_nm": worst_gravity.gravity_coefficient_nm(),
            "gravity_torque_at_target_nm": worst_gravity.gravity_torque_nm(
                config.requirements.target_recovery_angle_deg
            ),
            "swing_up_energy_j": worst_gravity.swing_up_energy_j(),
            "fastest_upright_time_constant_ms": fastest_time_constant * 1000.0,
            "preliminary_maximum_delay_ms": (
                fastest_time_constant
                * config.requirements.maximum_delay_fraction_of_time_constant
                * 1000.0
            ),
            "preliminary_minimum_sample_rate_hz": 1.0
            / (
                fastest_time_constant
                * config.requirements.maximum_sample_period_fraction_of_time_constant
            ),
            "minimum_flywheel_inertia_kg_m2": minimum_flywheel.flywheel_inertia_kg_m2,
        },
        "nominal_actuators": nominal_actuators,
        "robust_actuators": robust_actuators,
    }


def _number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    nominal = report["nominal_design"]
    worst = report["worst_case"]
    lines = [
        "# Phase 0 feasibility pre-check",
        "",
        f"**Gate status: {report['status']}** — {report['reason']}.",
        "",
        "## Mechanical envelope",
        "",
        "| Quantity | Nominal | Pessimistic |",
        "|---|---:|---:|",
        f"| Gravity coefficient G | {_number(nominal['gravity_coefficient_nm'])} Nm | {_number(worst['gravity_coefficient_nm'])} Nm |",
        f"| Gravity torque at target angle | {_number(nominal['gravity_torque_at_target_nm'])} Nm | {_number(worst['gravity_torque_at_target_nm'])} Nm |",
        f"| Down-to-up potential energy | {_number(nominal['swing_up_energy_j'])} J | {_number(worst['swing_up_energy_j'])} J |",
        f"| Upright time constant | {_number(nominal['upright_time_constant_ms'], 1)} ms | {_number(worst['fastest_upright_time_constant_ms'], 1)} ms (fastest) |",
        "",
        "## Actuator checks at the pessimistic gravity corner",
        "",
        "| Actuator | Peak torque | Margin at target | Safe angle (minimum margin) | Driver continuous proxy | Peak gate | Proxy gate |",
        "|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    for actuator in report["robust_actuators"]:
        lines.append(
            "| {name} | {peak} Nm | {margin}x | {angle} deg | {continuous} Nm | {peak_gate} | {continuous_gate} |".format(
                name=actuator["name"],
                peak=_number(actuator["peak_torque_nm"]),
                margin=_number(actuator["peak_margin_at_target_angle"], 2),
                angle=_number(
                    actuator["recoverable_angle_deg_at_minimum_margin"], 1
                ),
                continuous=_number(
                    actuator["driver_continuous_torque_proxy_nm"]
                ),
                peak_gate=actuator["peak_gate"],
                continuous_gate=actuator["driver_continuous_proxy_gate"],
            )
        )
    lines.extend(
        [
            "",
            "The continuous value is only the controller current rating multiplied by Kt. It is not evidence that the motor can dissipate that heat continuously.",
            "",
            "## Preliminary sensor timing envelope",
            "",
            f"- Minimum sample rate: {_number(worst['preliminary_minimum_sample_rate_hz'], 0)} Hz",
            f"- Maximum total delay: {_number(worst['preliminary_maximum_delay_ms'], 1)} ms",
            "- Angle/rate noise: PROVISIONAL in `phase0-control`; not validated for a selected sensor/estimator",
            "",
            "These timing values are time-scale heuristics, not a Phase 2 simulation result.",
            "",
            "## Remaining gate inputs",
            "",
            "- component-level mass, placement, and inertia from CAD or measurements",
            "- measured or manufacturer-supported mj5208 continuous torque at zero speed",
            "- cogging torque before and after moteus compensation",
            "- required swing-up momentum from the constrained nonlinear simulation",
            "- validation with the selected sensor/estimator, quantization, and jitter",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--format", choices=("markdown", "json"), default="markdown"
    )
    args = parser.parse_args()

    report = build_report(load_config(args.config))
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(render_markdown(report))


if __name__ == "__main__":
    main()
