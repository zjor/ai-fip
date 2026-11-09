"""Plot the PNDLM-13 thermal-assumption figure for the article draft.

Usage, from software/sim:

    poetry run python -m app.plot_thermal \
        phase0.toml \
        ../../docs/hardware/measurements/2026-10-26-torque-constant.json \
        ../../docs/articles/figures/2026-11-09-thermal-assumption.png

No measurement: the figure shows the steady-state copper-loss model of
phase0.toml with the measured K_t and the standard temperature coefficients
below. Steady state: ΔT = R_th · 1.5 · I² · R(T), R(T) = R0 · (1 + α_Cu·ΔT).

Top panel: continuous torque at the allowed winding rise against the
winding-to-ambient thermal resistance R_th, cold K_t and hot K_t; the
assumed R_th band, the requirement and the break-even R_th.
Bottom panel: steady winding rise against phase current for the assumed
R_th values and the break-even value, with the allowed rise, the current for
the requirement at hot K_t, and the swing-up RMS current band (short events,
not steady state).

Prints the numbers the article quotes.
"""

from __future__ import annotations

import json
import math
import sys
import tomllib
from pathlib import Path

from app.plot_style import (
    FAIL,
    FAIL_BAND,
    INK_2,
    MEASURED,
    MEASURED_2,
    MUTED,
    REFERENCE,
    ZONE,
    footer,
    header,
    layout,
    plt,
    style,
)

ALPHA_CU = 0.00393             # 1/K, copper resistance temperature coefficient
MAGNET_TC = (-0.0010, -0.0012)  # 1/K, NdFeB remanence, typical range (grade unknown)
SWING_UP_RMS_A = (5.0, 11.0)   # docs/physics/phase-0-feasibility.md, about 1 s


def steady_rise(i: float, r_th: float, r0: float) -> float:
    """Solve ΔT = r_th·1.5·i²·r0·(1 + α·ΔT) for ΔT; inf if it runs away."""
    k = r_th * 1.5 * i * i * r0
    den = 1.0 - k * ALPHA_CU
    return k / den if den > 0 else math.inf


def continuous_current(r_th: float, r0: float, rise: float) -> float:
    return math.sqrt(rise / (r_th * 1.5 * r0 * (1.0 + ALPHA_CU * rise)))


def main(toml_path: Path, kt_path: Path, target: Path) -> None:
    cfg = tomllib.loads(toml_path.read_text())
    motor = cfg["motor"]
    r0 = motor["phase_resistance_ohm"]
    r_ths = motor["thermal_resistance_k_per_w"]
    rise = motor["allowed_winding_rise_k"]
    need_nm = cfg["requirements"]["minimum_continuous_torque_nm"]
    kt = json.loads(kt_path.read_text())["simulator"]["kt_linear_region_per_commanded_a_nm_per_a"]

    kt_hot = [kt * (1.0 + tc * rise) for tc in MAGNET_TC]
    kt_hot_low = min(kt_hot)
    i_need = need_nm / kt_hot_low
    p_need = 1.5 * i_need ** 2 * r0 * (1.0 + ALPHA_CU * rise)
    r_th_break = rise / p_need

    print(f"R0 {r0} ohm -> hot {r0 * (1 + ALPHA_CU * rise):.4f} ohm (+{ALPHA_CU * rise * 100:.1f} %)")
    print(f"K_t {kt:.4f} -> hot {kt_hot[0]:.4f}..{kt_hot[1]:.4f}")
    for r_th in r_ths + [r_th_break]:
        i = continuous_current(r_th, r0, rise)
        print(f"R_th {r_th:.2f}: I_cont {i:.2f} A, torque hot {i * kt_hot_low:.3f}..{i * max(kt_hot):.3f}"
              f" Nm, cold-K_t {i * kt:.3f} Nm")
    print(f"requirement {need_nm} Nm -> {i_need:.2f} A hot, P {p_need:.2f} W, break-even R_th {r_th_break:.2f} K/W")
    for r_th in r_ths:
        print(f"R_th {r_th}: rise at {i_need:.2f} A = {steady_rise(i_need, r_th, r0):.1f} K")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.0, 9.2))

    xs = [1.0 + 0.05 * k for k in range(0, 221)]
    top.axvspan(min(r_ths), max(r_ths), color=ZONE, lw=0, label=f"допущение: {min(r_ths):g}–{max(r_ths):g} K/W")
    top.axvspan(r_th_break, 12.0, color=FAIL_BAND, alpha=0.5, lw=0, label="требование не выполняется")
    top.plot(xs, [continuous_current(x, r0, rise) * kt for x in xs], color=REFERENCE, lw=1.4, ls="--",
             label=f"K_t холодный {kt:.4f} Nm/A")
    top.plot(xs, [continuous_current(x, r0, rise) * kt_hot_low for x in xs], color=MEASURED, lw=2.2,
             label=f"K_t горячий {kt_hot_low:.4f} Nm/A (магниты −{-MAGNET_TC[1] * rise * 100:.0f} %)")
    top.axhline(need_nm, color=FAIL, lw=1.2, label=f"требование {need_nm:.2f} Nm")
    top.plot([r_th_break], [need_nm], "o", ms=8, color=FAIL)
    top.annotate(f"{r_th_break:.1f} K/W", (r_th_break, need_nm), xytext=(8, 10), textcoords="offset points",
                 color=FAIL, fontsize=11, weight="bold")
    top.set_xlim(1.0, 12.0)
    top.set_ylim(0.0, 0.6)
    top.set_xlabel("тепловое сопротивление обмотка → воздух R_th, K/W")
    top.set_ylabel("длительный момент, Nm")
    top.legend(loc="upper right", frameon=False, fontsize=10)
    style(top)

    ii = [0.1 * k for k in range(0, 181)]
    colors = {r_ths[0]: REFERENCE, r_ths[1]: MEASURED, r_ths[2]: MEASURED_2}
    for r_th in r_ths:
        bottom.plot(ii, [min(steady_rise(i, r_th, r0), 200) for i in ii], color=colors.get(r_th, MEASURED),
                    lw=2.0, label=f"R_th = {r_th:g} K/W")
    bottom.plot(ii, [min(steady_rise(i, r_th_break, r0), 200) for i in ii], color=FAIL, lw=1.4, ls="--",
                label=f"R_th = {r_th_break:.1f} K/W (граница)")
    bottom.axhline(rise, color=REFERENCE, lw=1.2, ls=":", label=f"допустимый нагрев {rise:g} K")
    bottom.axvline(i_need, color=FAIL, lw=1.0)
    bottom.annotate(f"{need_nm:.2f} Nm = {i_need:.1f} A", (i_need, 105), xytext=(4, 0), textcoords="offset points",
                    color=FAIL, fontsize=10)
    bottom.axvspan(*SWING_UP_RMS_A, ymin=0, ymax=0.04, color=MUTED, alpha=0.6, lw=0)
    bottom.annotate("ток раскачки, ~1 с", (SWING_UP_RMS_A[1], 2), xytext=(6, -2),
                    textcoords="offset points", color=INK_2, fontsize=9.5)
    bottom.set_xlim(0, 18)
    bottom.set_ylim(0, 120)
    bottom.set_xlabel("постоянный фазный ток, A")
    bottom.set_ylabel("установившийся нагрев обмотки, K")
    bottom.legend(loc="upper left", frameon=False, fontsize=10)
    style(bottom)

    header(fig, f"Нагрев: требование 0.20 Nm выполняется до R_th ≈ {r_th_break:.0f} K/W",
           "Модель потерь в меди с нагревом R и ослаблением магнитов; измерения температуры нет")
    footer(fig, "PNDLM-13 · допущение, не измерение · phase0.toml [motor], "
                "K_t — measurements/2026-10-26-torque-constant.json")
    layout(fig)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    print(f"wrote {target}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
