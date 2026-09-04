"""Analytical pre-check for the Phase 0 hardware feasibility gate.

This module intentionally keeps unknowns visible. In particular, a driver
continuous-current rating is not treated as proof of the motor's thermal
continuous torque, and flywheel momentum is not marked as sufficient until the
swing-up simulation (``app.phase0_swingup``) provides the required momentum.

Motor electrical conventions follow the moteus firmware so that the numbers
here match what the controller will report and enforce:

* ``Kt = (3/2) * (1/sqrt(3)) * (60 / 2 pi) / Kv``  (fw/bldc_servo.cc)
* ``v_per_hz = 60 / (sqrt(3) * Kv)``               (fw/bldc_servo_control.h)
* ``V_eff = 0.5 * V_bus * max_voltage_ratio * (1 - modulation_margin)``
* voltage circle at rotor frequency ``f`` and q-axis current ``iq``:
  ``(v_per_hz f + R iq)^2 + (pi poles L iq f)^2 <= V_eff^2``
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
MOTEUS_TORQUE_FACTOR = (3.0 / 2.0) * (1.0 / math.sqrt(3.0)) * (60.0 / (2.0 * math.pi))
RAD_S_PER_RPM = 2.0 * math.pi / 60.0


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
class Battery:
    mass_kg: float
    top_end_radius_m: float


@dataclass(frozen=True)
class Requirements:
    target_recovery_angle_deg: float
    preferred_recovery_angle_deg: float
    minimum_peak_margin: float
    preferred_peak_margin: float
    minimum_continuous_torque_nm: float
    maximum_delay_fraction_of_time_constant: float
    maximum_sample_period_fraction_of_time_constant: float
    maximum_wheel_speed_fraction_at_catch: float


@dataclass(frozen=True)
class Moteus:
    modulation_margin: float
    max_voltage_ratio: float

    def effective_phase_voltage_v(self, bus_voltage_v: float) -> float:
        return 0.5 * bus_voltage_v * self.max_voltage_ratio * (1.0 - self.modulation_margin)


@dataclass(frozen=True)
class Motor:
    name: str
    kv_rpm_per_v: float
    kv_measured_rpm_per_v: float
    phase_resistance_ohm: float
    phase_inductance_h: float
    poles: int
    peak_torque_nm: float
    max_rpm: float
    mass_kg: float
    thermal_resistance_k_per_w: Range3
    allowed_winding_rise_k: float

    def torque_constant_nm_per_a(self, kv_rpm_per_v: float | None = None) -> float:
        return MOTEUS_TORQUE_FACTOR / (kv_rpm_per_v or self.kv_rpm_per_v)

    def v_per_hz(self, kv_rpm_per_v: float | None = None) -> float:
        return 60.0 / (math.sqrt(3.0) * (kv_rpm_per_v or self.kv_rpm_per_v))

    def copper_loss_w(self, peak_phase_current_a: float) -> float:
        return 1.5 * peak_phase_current_a**2 * self.phase_resistance_ohm

    def thermal_continuous_current_a(self, thermal_resistance_k_per_w: float) -> float:
        power = self.allowed_winding_rise_k / thermal_resistance_k_per_w
        return math.sqrt(power / (1.5 * self.phase_resistance_ohm))


@dataclass(frozen=True)
class Driver:
    name: str
    peak_phase_current_a: float
    continuous_phase_current_a: float
    continuous_phase_current_cooled_a: float
    max_bus_voltage_v: float
    mass_kg: float
    price_usd: float


@dataclass(frozen=True)
class Actuator:
    """A motor + driver pair evaluated with the moteus firmware conventions."""

    motor: Motor
    driver: Driver
    moteus: Moteus

    @property
    def name(self) -> str:
        return f"{self.motor.name} + {self.driver.name}"

    @property
    def motor_kv_rpm_per_v(self) -> float:
        return self.motor.kv_rpm_per_v

    @property
    def motor_kt_nm_per_a(self) -> float:
        return self.motor.torque_constant_nm_per_a()

    def peak_torque_nm(self) -> float:
        return min(
            self.motor_kt_nm_per_a * self.driver.peak_phase_current_a,
            self.motor.peak_torque_nm,
        )

    def driver_continuous_torque_proxy_nm(self) -> float:
        return min(
            self.motor_kt_nm_per_a * self.driver.continuous_phase_current_a,
            self.motor.peak_torque_nm,
        )

    def motor_thermal_continuous_torque_nm(self, thermal_resistance_k_per_w: float) -> float:
        current = min(
            self.motor.thermal_continuous_current_a(thermal_resistance_k_per_w),
            self.driver.continuous_phase_current_cooled_a,
        )
        return min(self.motor_kt_nm_per_a * current, self.motor.peak_torque_nm)

    def max_q_current_a(self, bus_voltage_v: float, rotor_speed_rad_s: float) -> float:
        """Largest q-axis current inside the voltage circle at this speed."""
        f = abs(rotor_speed_rad_s) / (2.0 * math.pi)
        v_eff = self.moteus.effective_phase_voltage_v(bus_voltage_v)
        r = self.motor.phase_resistance_ohm
        v_per_hz = self.motor.v_per_hz()
        omega_l = math.pi * self.motor.poles * self.motor.phase_inductance_h * f
        a = r * r + omega_l * omega_l
        b = 2.0 * r * v_per_hz * f
        c = (v_per_hz * f) ** 2 - v_eff * v_eff
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return 0.0
        current = (-b + math.sqrt(disc)) / (2.0 * a)
        return max(0.0, current)

    def available_torque_nm(self, bus_voltage_v: float, rotor_speed_rad_s: float) -> float:
        current = min(
            self.max_q_current_a(bus_voltage_v, rotor_speed_rad_s),
            self.driver.peak_phase_current_a,
        )
        return min(self.motor_kt_nm_per_a * current, self.motor.peak_torque_nm)

    def no_load_rpm(self, bus_voltage_v: float) -> float:
        v_eff = self.moteus.effective_phase_voltage_v(bus_voltage_v)
        rpm = v_eff / self.motor.v_per_hz() * 60.0
        return min(rpm, self.motor.max_rpm)

    def knee_rpm(self, bus_voltage_v: float) -> float:
        """Speed above which peak torque starts to fall (bisection on the envelope)."""
        peak = self.peak_torque_nm()
        low, high = 0.0, self.no_load_rpm(bus_voltage_v) * RAD_S_PER_RPM
        if self.available_torque_nm(bus_voltage_v, 0.0) < peak * 0.999:
            return 0.0
        for _ in range(60):
            mid = 0.5 * (low + high)
            if self.available_torque_nm(bus_voltage_v, mid) >= peak * 0.999:
                low = mid
            else:
                high = mid
        return low / RAD_S_PER_RPM


@dataclass(frozen=True)
class Config:
    design: Design
    battery: Battery
    requirements: Requirements
    minimum_bus_voltage_v: float
    nominal_bus_voltage_v: float
    motor: Motor
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
        center_of_mass_m=Range3.from_list(design_raw["center_of_mass_m"], "center_of_mass_m"),
        pivot_inertia_kg_m2=Range3.from_list(
            design_raw["pivot_inertia_kg_m2"], "pivot_inertia_kg_m2"
        ),
        flywheel_inertia_kg_m2=Range3.from_list(
            design_raw["flywheel_inertia_kg_m2"], "flywheel_inertia_kg_m2"
        ),
    )
    motor_raw = dict(raw["motor"])
    motor_raw["thermal_resistance_k_per_w"] = Range3.from_list(
        motor_raw["thermal_resistance_k_per_w"], "thermal_resistance_k_per_w"
    )
    motor = Motor(**motor_raw)
    moteus = Moteus(**raw["moteus"])
    drivers = tuple(Driver(**item) for item in raw["drivers"])
    return Config(
        design=design,
        battery=Battery(**raw["battery"]),
        requirements=Requirements(**raw["requirements"]),
        minimum_bus_voltage_v=float(raw["power"]["minimum_bus_voltage_v"]),
        nominal_bus_voltage_v=float(raw["power"]["nominal_bus_voltage_v"]),
        motor=motor,
        actuators=tuple(Actuator(motor, driver, moteus) for driver in drivers),
    )


def nominal_point(design: Design) -> DesignPoint:
    return DesignPoint(
        design.total_mass_kg.nominal,
        design.center_of_mass_m.nominal,
        design.pivot_inertia_kg_m2.nominal,
        design.flywheel_inertia_kg_m2.nominal,
    )


def pessimistic_point(design: Design) -> DesignPoint:
    """Heaviest, longest, lightest-inertia corner: hardest to hold and to catch."""
    return DesignPoint(
        design.total_mass_kg.high,
        design.center_of_mass_m.high,
        design.pivot_inertia_kg_m2.low,
        design.flywheel_inertia_kg_m2.low,
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
    return flywheel_inertia_kg_m2 * speed_rpm * RAD_S_PER_RPM


def battery_placement(point: DesignPoint, battery: Battery) -> dict[str, Any]:
    """Compare adding the pack on the pivot axis vs. at the top end of the rod."""

    def variant(radius_m: float) -> dict[str, float]:
        gravity = point.gravity_coefficient_nm() + battery.mass_kg * GRAVITY_M_S2 * radius_m
        inertia = point.pivot_inertia_kg_m2 + battery.mass_kg * radius_m**2
        return {
            "gravity_coefficient_nm": gravity,
            "gravity_torque_at_20deg_nm": gravity * math.sin(math.radians(20.0)),
            "swing_up_energy_j": 2.0 * gravity,
            "pivot_inertia_kg_m2": inertia,
            "upright_time_constant_ms": 1000.0 * math.sqrt(inertia / gravity),
        }

    return {
        "battery_mass_kg": battery.mass_kg,
        "without_battery": variant(0.0),
        "on_pivot_axis": variant(0.0),
        "at_top_end": variant(battery.top_end_radius_m),
    }


def actuator_result(
    actuator: Actuator,
    point: DesignPoint,
    requirements: Requirements,
    bus_voltage_v: float,
) -> dict[str, Any]:
    peak_torque = actuator.peak_torque_nm()
    gravity_target = point.gravity_torque_nm(requirements.target_recovery_angle_deg)
    gravity_preferred = point.gravity_torque_nm(requirements.preferred_recovery_angle_deg)
    no_load_rpm = actuator.no_load_rpm(bus_voltage_v)
    thermal = actuator.motor.thermal_resistance_k_per_w
    return {
        "name": actuator.name,
        "torque_constant_nm_per_a": actuator.motor_kt_nm_per_a,
        "peak_torque_nm": peak_torque,
        "peak_current_for_motor_peak_a": actuator.motor.peak_torque_nm / actuator.motor_kt_nm_per_a,
        "driver_continuous_torque_proxy_nm": actuator.driver_continuous_torque_proxy_nm(),
        "motor_thermal_continuous_torque_nm": {
            "optimistic": actuator.motor_thermal_continuous_torque_nm(thermal.low),
            "nominal": actuator.motor_thermal_continuous_torque_nm(thermal.nominal),
            "pessimistic": actuator.motor_thermal_continuous_torque_nm(thermal.high),
        },
        "no_load_rpm": no_load_rpm,
        "knee_rpm": actuator.knee_rpm(bus_voltage_v),
        "stall_torque_at_bus_nm": actuator.available_torque_nm(bus_voltage_v, 0.0),
        "peak_margin_at_target_angle": peak_torque / gravity_target,
        "peak_margin_at_preferred_angle": peak_torque / gravity_preferred,
        "recoverable_angle_deg_at_minimum_margin": recoverable_angle_deg(
            peak_torque, point.gravity_coefficient_nm(), requirements.minimum_peak_margin
        ),
        "flywheel_momentum_capacity_nms": angular_momentum_nms(
            point.flywheel_inertia_kg_m2, no_load_rpm
        ),
    }


def _gate(value: float, threshold: float) -> str:
    return "PASS" if value >= threshold else "FAIL"


def build_report(config: Config) -> dict[str, Any]:
    nominal = nominal_point(config.design)
    corners = design_corners(config.design)
    worst_gravity = max(corners, key=lambda point: point.gravity_coefficient_nm())
    fastest_instability = min(corners, key=lambda point: point.upright_time_constant_s())
    minimum_flywheel = min(corners, key=lambda point: point.flywheel_inertia_kg_m2)
    requirements = config.requirements

    nominal_actuators = [
        actuator_result(actuator, nominal, requirements, config.minimum_bus_voltage_v)
        for actuator in config.actuators
    ]
    robust_actuators = []
    for actuator in config.actuators:
        result = actuator_result(actuator, worst_gravity, requirements, config.minimum_bus_voltage_v)
        result["peak_gate"] = _gate(
            result["peak_margin_at_target_angle"], requirements.minimum_peak_margin
        )
        result["preferred_peak_gate"] = _gate(
            result["peak_margin_at_preferred_angle"], requirements.minimum_peak_margin
        )
        result["driver_continuous_proxy_gate"] = _gate(
            result["driver_continuous_torque_proxy_nm"],
            requirements.minimum_continuous_torque_nm,
        )
        result["motor_thermal_gate"] = (
            _gate(
                result["motor_thermal_continuous_torque_nm"]["pessimistic"],
                requirements.minimum_continuous_torque_nm,
            )
            + " (model, unverified)"
        )
        result["swing_up_momentum_gate"] = "see phase0-swingup"
        robust_actuators.append(result)

    fastest_time_constant = fastest_instability.upright_time_constant_s()
    return {
        "status": "NOT READY",
        "reason": (
            "mass distribution is a planning range, motor thermal model is unverified, "
            "cogging is unknown, and the sensor/estimator is not selected"
        ),
        "nominal_design": {
            **asdict(nominal),
            "gravity_coefficient_nm": nominal.gravity_coefficient_nm(),
            "gravity_torque_at_target_nm": nominal.gravity_torque_nm(
                requirements.target_recovery_angle_deg
            ),
            "gravity_torque_at_preferred_nm": nominal.gravity_torque_nm(
                requirements.preferred_recovery_angle_deg
            ),
            "swing_up_energy_j": nominal.swing_up_energy_j(),
            "upright_time_constant_ms": nominal.upright_time_constant_s() * 1000.0,
        },
        "worst_case": {
            "gravity_coefficient_nm": worst_gravity.gravity_coefficient_nm(),
            "gravity_torque_at_target_nm": worst_gravity.gravity_torque_nm(
                requirements.target_recovery_angle_deg
            ),
            "gravity_torque_at_preferred_nm": worst_gravity.gravity_torque_nm(
                requirements.preferred_recovery_angle_deg
            ),
            "swing_up_energy_j": worst_gravity.swing_up_energy_j(),
            "fastest_upright_time_constant_ms": fastest_time_constant * 1000.0,
            "preliminary_maximum_delay_ms": (
                fastest_time_constant
                * requirements.maximum_delay_fraction_of_time_constant
                * 1000.0
            ),
            "preliminary_minimum_sample_rate_hz": 1.0
            / (fastest_time_constant * requirements.maximum_sample_period_fraction_of_time_constant),
            "minimum_flywheel_inertia_kg_m2": minimum_flywheel.flywheel_inertia_kg_m2,
        },
        "battery_placement": battery_placement(nominal, config.battery),
        "nominal_actuators": nominal_actuators,
        "robust_actuators": robust_actuators,
    }


def _number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    nominal = report["nominal_design"]
    worst = report["worst_case"]
    battery = report["battery_placement"]
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
        f"| Gravity torque at preferred angle | {_number(nominal['gravity_torque_at_preferred_nm'])} Nm | {_number(worst['gravity_torque_at_preferred_nm'])} Nm |",
        f"| Down-to-up potential energy | {_number(nominal['swing_up_energy_j'])} J | {_number(worst['swing_up_energy_j'])} J |",
        f"| Upright time constant | {_number(nominal['upright_time_constant_ms'], 1)} ms | {_number(worst['fastest_upright_time_constant_ms'], 1)} ms (fastest) |",
        "",
        "## Battery placement (nominal design, pack mass "
        f"{_number(battery['battery_mass_kg'], 2)} kg)",
        "",
        "| Placement | G | τ_g at 20° | Swing-up energy | I_p | t_c |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("no battery", "without_battery"), ("on pivot axis", "on_pivot_axis"), ("at top end", "at_top_end")):
        item = battery[key]
        lines.append(
            f"| {label} | {_number(item['gravity_coefficient_nm'])} Nm | {_number(item['gravity_torque_at_20deg_nm'])} Nm | "
            f"{_number(item['swing_up_energy_j'], 2)} J | {_number(item['pivot_inertia_kg_m2'], 4)} kg m² | {_number(item['upright_time_constant_ms'], 0)} ms |"
        )
    lines.extend(
        [
            "",
            "## Actuator checks at the pessimistic gravity corner, minimum bus voltage",
            "",
            "| Actuator | Kt | Peak torque | Margin at target / preferred | Safe angle (min margin) | Driver continuous proxy | Motor thermal continuous (model, opt/nom/pess) | No-load / knee rpm | Peak gate | Proxy gate | Thermal gate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|",
        ]
    )
    for actuator in report["robust_actuators"]:
        thermal = actuator["motor_thermal_continuous_torque_nm"]
        lines.append(
            "| {name} | {kt} Nm/A | {peak} Nm | {margin}x / {margin_pref}x | {angle} deg | {continuous} Nm | {t_opt} / {t_nom} / {t_pess} Nm | {nl} / {knee} | {peak_gate} | {continuous_gate} | {thermal_gate} |".format(
                name=actuator["name"],
                kt=_number(actuator["torque_constant_nm_per_a"], 4),
                peak=_number(actuator["peak_torque_nm"]),
                margin=_number(actuator["peak_margin_at_target_angle"], 2),
                margin_pref=_number(actuator["peak_margin_at_preferred_angle"], 2),
                angle=_number(actuator["recoverable_angle_deg_at_minimum_margin"], 1),
                continuous=_number(actuator["driver_continuous_torque_proxy_nm"]),
                t_opt=_number(thermal["optimistic"], 2),
                t_nom=_number(thermal["nominal"], 2),
                t_pess=_number(thermal["pessimistic"], 2),
                nl=_number(actuator["no_load_rpm"], 0),
                knee=_number(actuator["knee_rpm"], 0),
                peak_gate=actuator["peak_gate"],
                continuous_gate=actuator["driver_continuous_proxy_gate"],
                thermal_gate=actuator["motor_thermal_gate"],
            )
        )
    lines.extend(
        [
            "",
            "Kt uses the moteus firmware convention 8.27/Kv with the nominal Kv 330 (measured devkit Kv 304 gives 9% more torque per amp). "
            "The driver proxy is the controller current rating times Kt. The motor thermal value is a copper-loss model "
            "(P = 1.5 I² R, R = 0.047 Ω) with an assumed winding-to-ambient thermal resistance; it is not a measurement.",
            "",
            "## Preliminary sensor timing envelope",
            "",
            f"- Minimum sample rate: {_number(worst['preliminary_minimum_sample_rate_hz'], 0)} Hz",
            f"- Maximum total delay: {_number(worst['preliminary_maximum_delay_ms'], 1)} ms",
            "- Angle/rate noise: PROVISIONAL in `phase0-control`; not validated for a selected sensor/estimator",
            "",
            "## Remaining gate inputs",
            "",
            "- component-level mass, placement, and inertia from CAD or measurements",
            "- bench confirmation of the mj5208 thermal model (stall current vs. winding temperature)",
            "- cogging torque before and after moteus anti-cogging",
            "- swing-up momentum: run `poetry run phase0-swingup` (results in docs/physics/phase-0-feasibility.md)",
            "- validation with the selected sensor/estimator, quantization, and jitter",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()

    report = build_report(load_config(args.config))
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(render_markdown(report))


if __name__ == "__main__":
    main()
