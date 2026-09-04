#!/usr/bin/env python3
"""Pendulum mass model from rendered STLs plus purchased hardware.

Printed parts come from build/*.stl (pendulum frame; wheel parts in the wheel
frame at z = rod_len). Hardware masses are catalogue values. Output feeds
software/sim/phase0.toml.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stl_props  # noqa: E402

G = 9.81
RHO_PLA = 1.25e-3
RHO_STEEL = 7.85e-3
ROD_LEN_MM = 250.0

# name: (mass_g, z_com_mm). Sources: mjbots pages (motor, r4.11), typical 4S 1000 mAh
# pack, RPi 4 + pi3hat + standoffs (unsure, +-15 g), M3/M8 hardware estimate.
HARDWARE = {
    "mj5208": (193.0, ROD_LEN_MM),
    "moteus_r4.11": (14.2, ROD_LEN_MM),
    "motor_screws": (5.0, ROD_LEN_MM),
    "battery_4s": (120.0, 0.0),
    "rpi4_pi3hat": (85.0, 0.0),
    "axle_collars_clamp_screws": (60.0, 0.0),
}


def _stl(build: Path, name: str, axis: str):
    return stl_props.props(stl_props.read_stl(build / f"{name}.stl"), axis)


def compute(build: Path) -> dict:
    parts: list[tuple[str, float, float]] = []
    quadrant = _stl(build, "wheel_quadrant", "z")
    hub = _stl(build, "wheel_hub", "z")
    bolts = _stl(build, "wheel_bolts", "z")
    wheel_plastic_g = (4 * quadrant["volume_mm3"] + hub["volume_mm3"]) * RHO_PLA
    bolts_g = bolts["volume_mm3"] * RHO_STEEL
    i_w = (
        (4 * quadrant["i_axis_mm5"] + hub["i_axis_mm5"]) * RHO_PLA
        + bolts["i_axis_mm5"] * RHO_STEEL
    ) * 1e-9
    parts.append(("wheel_plastic", wheel_plastic_g, ROD_LEN_MM))
    parts.append(("wheel_bolts", bolts_g, ROD_LEN_MM))
    for name in ("beam", "motor_flange"):
        p = _stl(build, name, "y")
        parts.append((name, p["volume_mm3"] * RHO_PLA, p["centroid"][2]))
    for name, (mass, z) in HARDWARE.items():
        parts.append((name, mass, z))

    m_t = sum(m for _, m, _ in parts) / 1000.0
    first = sum(m / 1000.0 * z / 1000.0 for _, m, z in parts)
    l_c = first / m_t
    # pendulum body inertia about the pivot: point masses at their CoM plus the
    # beam as a slender rod (m L^2 / 3 about its lower end)
    i_p = sum(m / 1000.0 * (z / 1000.0) ** 2 for name, m, z in parts if name != "beam")
    beam_mass = next(m for name, m, _ in parts if name == "beam") / 1000.0
    i_p += beam_mass * (ROD_LEN_MM / 1000.0) ** 2 / 3.0
    return {
        "m_t_kg": m_t,
        "l_c_m": l_c,
        "G_nm": m_t * G * l_c,
        "I_p_kg_m2": i_p,
        "I_w_kg_m2": i_w,
        "parts": parts,
    }


def main() -> None:
    build = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "build"
    model = compute(build)
    print("| Part | Mass g | z CoM mm |")
    print("|---|---:|---:|")
    for name, mass, z in model["parts"]:
        print(f"| {name} | {mass:.1f} | {z:.0f} |")
    print()
    print(f"m_t = {model['m_t_kg']:.3f} kg, l_c = {model['l_c_m']:.3f} m, G = {model['G_nm']:.3f} Nm")
    print(f"I_p = {model['I_p_kg_m2']:.4f} kg m2, I_w = {model['I_w_kg_m2']:.4f} kg m2")
    print()
    print("phase0.toml [design] suggestion (nominal, -10%/+10%):")
    for key, value in (
        ("total_mass_kg", model["m_t_kg"]),
        ("center_of_mass_m", model["l_c_m"]),
        ("pivot_inertia_kg_m2", model["I_p_kg_m2"]),
        ("flywheel_inertia_kg_m2", model["I_w_kg_m2"]),
    ):
        print(f"{key} = [{value * 0.9:.4g}, {value:.4g}, {value * 1.1:.4g}]")


if __name__ == "__main__":
    main()
