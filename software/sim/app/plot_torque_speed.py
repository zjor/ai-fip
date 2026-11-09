"""Plot the PNDLM-12 torque-speed figures for the article draft.

Usage, from software/sim:

    poetry run python -m app.plot_torque_speed \
        ../../docs/hardware/measurements/2026-11-04-torque-speed.json \
        ../../docs/articles/figures/2026-11-04-torque-speed.png \
        ../../docs/articles/figures/2026-11-04-voltage-circle.png

Figure 1 (torque-speed), two panels:
- top: no-load speed against the measured bus, every saturated step, the
  fitted line, the old Phase 0 model (Kv 330, r_max 0.94), the hobby rule
  Kv·V_bus, and the 16.0 V velocity-capped steps as lower bounds;
- bottom: torque against speed at the 12.0 V run (bus 11.81 V): the voltage
  limit K_t·i_max(ω) of the new and the old model, the torque available at the
  three step currents, and the measured base-speed and no-load points.

Figure 2 (voltage circle): the phase-voltage vectors of the moteus model at
11.81 V and 5.28 A, at 1500 rpm (fits) and 2900 rpm (does not fit), each
with a zoom on the vector tip.

Model: v_per_hz = 60/(sqrt3·Kv), V_eff = 0.5·V_bus·r_max·(1 − m),
(v_per_hz·f + R·i)² + (π·p·L·i·f)² ≤ V_eff², f in rev/s; R, L, p, m from the
record's r_max inputs. Torque is K_t·i_q, losses not subtracted.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from matplotlib.patches import Circle, FancyArrowPatch

from app.plot_style import (
    FAIL,
    INK,
    INK_2,
    MEASURED,
    MEASURED_2,
    MUTED,
    REFERENCE,
    footer,
    header,
    layout,
    plt,
    style,
)

SQRT3 = math.sqrt(3.0)
OLD_KV = 330.0
OLD_R_MAX = 0.94
PANEL_RUN_SET_V = 12.0
STEP_CURRENTS_A = (1.77, 3.50, 5.28)  # mean low-speed q-current of the 12.0 V steps
CIRCLE_BUS_V = 11.81
CIRCLE_CURRENT_A = 5.28
CIRCLE_SPEEDS_RPM = (1500.0, 2900.0)


class Model:
    def __init__(self, kv: float, r_max: float, inputs: dict) -> None:
        self.v = 60.0 / (SQRT3 * kv)
        self.r_max = r_max
        self.r = inputs["r_ohm"]
        self.l = inputs["l_h"]
        self.p = inputs["poles"]
        self.m = inputs["m"]

    def v_eff(self, bus: float) -> float:
        return 0.5 * bus * self.r_max * (1.0 - self.m)

    def no_load_rpm(self, bus: float) -> float:
        return self.v_eff(bus) / self.v * 60.0

    def parts(self, rpm: float, i: float) -> tuple[float, float, float]:
        f = rpm / 60.0
        return self.v * f, self.r * i, math.pi * self.p * self.l * f * i

    def i_max(self, rpm: float, bus: float) -> float:
        f = rpm / 60.0
        a, b, ve = self.v * f, math.pi * self.p * self.l * f, self.v_eff(bus)
        aa = self.r ** 2 + b ** 2
        disc = (a * self.r) ** 2 - aa * (a * a - ve * ve)
        if disc <= 0:
            return 0.0
        return max(0.0, (-a * self.r + math.sqrt(disc)) / aa)


def torque_speed(record: dict, new: Model, old: Model, kt: float, target: Path) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.0, 9.4), height_ratios=(1.0, 1.15))

    line = record["no_load_line"]
    xs, ys, caps = [], [], []
    for run in record["runs"]:
        for s in run["steps"]:
            if s["lower_bound"]:
                caps.append((run["bus_v"], s["no_load_rpm"]))
            else:
                xs.append(s["no_load_bus_v"])
                ys.append(s["no_load_rpm"])
    grid_v = [10.0 + 0.05 * k for k in range(0, 141)]
    kv = 60.0 / (SQRT3 * new.v)
    top.plot(grid_v, [kv * v for v in grid_v],
             color=MUTED, lw=1.2, ls=":", label="«Kv × U» — 339 rpm/V, наивное правило")
    top.plot(grid_v, [old.no_load_rpm(v) for v in grid_v], color=REFERENCE, lw=1.4, ls="--",
             label=f"старая модель Phase 0: Kv 330, r_max 0.94 → {old.no_load_rpm(1.0):.0f} rpm/V")
    top.plot(grid_v, [line["intercept_rpm"] + line["slope_rpm_per_v"] * v for v in grid_v],
             color=MEASURED, lw=2.0,
             label=f"прямая по точкам: {line['slope_rpm_per_v']:.1f} ± {line['slope_se_rpm_per_v']:.1f} rpm/V")
    top.plot(xs, ys, "o", ms=5, color=MEASURED, label=f"холостая скорость, {len(xs)} точки")
    top.plot([c[0] for c in caps], [c[1] for c in caps], "^", ms=7, mfc="none", mec=MEASURED_2,
             mew=1.5, label="16.0 В: упор в останов 3822 rpm, нижняя граница")
    top.set_xlim(10.0, 17.0)
    top.set_ylim(2000, 6000)
    top.set_xlabel("напряжение питания под нагрузкой, В")
    top.set_ylabel("холостая скорость, rpm")
    top.legend(loc="upper left", frameon=False, fontsize=10)
    style(top)

    run = next(r for r in record["runs"] if r["bus_v"] == PANEL_RUN_SET_V)
    bus = sum(s["no_load_bus_v"] for s in run["steps"]) / len(run["steps"])
    grid = [5.0 * k for k in range(0, 701)]
    top_nm = 0.16
    bottom.plot(grid, [min(top_nm, kt * old.i_max(w, bus)) for w in grid], color=REFERENCE, lw=1.4,
                ls="--", label="предел по напряжению, старая модель")
    bottom.plot(grid, [min(top_nm, kt * new.i_max(w, bus)) for w in grid], color=MEASURED, lw=2.0,
                label="предел по напряжению, Kv 339, r_max 1.00")
    for k, i in enumerate(STEP_CURRENTS_A):
        bottom.plot(grid, [kt * min(i, new.i_max(w, bus)) for w in grid], color=INK_2, lw=1.0,
                    alpha=0.7, label="момент при токе шага 1.77 / 3.50 / 5.28 A" if k == 0 else None)
    bottom.plot([s["knee_rpm"] for s in run["steps"]],
                [0.9 * s["delivered_torque_nm"] for s in run["steps"]],
                "o", ms=6, color=MEASURED_2, label="базовая скорость (90 % момента), измерено")
    bottom.plot([s["no_load_rpm"] for s in run["steps"]], [0.002] * len(run["steps"]),
                "o", ms=6, color=MEASURED, label="холостая скорость, измерено")
    bottom.set_xlim(0, 3300)
    bottom.set_ylim(0, top_nm)
    bottom.set_xlabel("скорость, rpm")
    bottom.set_ylabel("момент K_t·i_q, Nm")
    bottom.legend(loc="lower left", frameon=False, fontsize=10)
    bottom.set_title(f"прогон 12.0 В (под нагрузкой {bus:.2f} В), обе стороны", loc="left",
                     fontsize=11.5, color=INK_2)
    style(bottom)

    header(fig, "Холостая скорость: 248 rpm/V, на 9 % выше старой модели",
           "Сверху — все прогоны против напряжения; снизу — момент против скорости при 12 В")
    footer(fig, "PNDLM-12 · 5 прогонов 20261104T111019Z…113421Z · "
                "measurements/2026-11-04-torque-speed.json")
    layout(fig)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    print(f"wrote {target}")


def _vectors(ax, model: Model, rpm: float, i: float, bus: float) -> bool:
    e, ri, li = model.parts(rpm, i)
    ve = model.v_eff(bus)
    need = math.hypot(e + ri, li)
    fits = need <= ve
    ring = MEASURED if fits else FAIL
    ax.add_patch(Circle((0, 0), ve, fill=True, fc=ring, alpha=0.06, lw=0))
    ax.add_patch(Circle((0, 0), ve, fill=False, ec=ring, lw=2.0))
    def arrow(x0, y0, x1, y1, color, lw, ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", color=color, lw=lw,
                                     ls=ls, shrinkA=0, shrinkB=0, mutation_scale=14))

    arrow(0, 0, 0, e, MEASURED, 2.4)
    arrow(0, e, 0, e + ri, MEASURED_2, 2.4)
    arrow(0, e + ri, -li, e + ri, INK, 2.4)
    arrow(0, 0, -li, e + ri, ring, 1.3, "--")
    return fits


def voltage_circle(new: Model, kt: float, target: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.9))
    bus, i = CIRCLE_BUS_V, CIRCLE_CURRENT_A
    ve = new.v_eff(bus)
    for ax, rpm in zip(axes, CIRCLE_SPEEDS_RPM):
        fits = _vectors(ax, new, rpm, i, bus)
        e, ri, li = new.parts(rpm, i)
        need = math.hypot(e + ri, li)
        ax.set_xlim(-5.6, 5.6)
        ax.set_ylim(-0.3, 6.6)
        ax.set_aspect("equal")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("ось d, В")
        if ax is axes[0]:
            ax.set_ylabel("ось q (даёт момент), В")
        ax.text(0.15, e * 0.45, f"E = {e:.2f} В", color=MEASURED, fontsize=10.5, weight="bold")
        verdict = "помещается" if fits else f"не помещается: i_max = {new.i_max(rpm, bus):.2f} A"
        ax.set_title(f"{rpm:.0f} rpm: нужно {need:.2f} В из {ve:.2f}\n{verdict}", loc="left",
                     fontsize=11, color=INK if fits else FAIL)
        style(ax)

        ins = ax.inset_axes([0.03, 0.50, 0.40, 0.36])
        _vectors(ins, new, rpm, i, bus)
        tip_y = e + ri
        ins.set_xlim(-0.75, 0.25)
        ins.set_ylim(tip_y - 0.55, tip_y + 0.35)
        ins.set_aspect("equal")
        ins.set_xticks([])
        ins.set_yticks([])
        ins.text(0.04, tip_y - ri / 2 - 0.03, "R·i", color=MEASURED_2, fontsize=9.5, weight="bold")
        ins.text(-li / 2 - 0.12, tip_y + 0.07, "ωL·i", color=INK, fontsize=9.5, weight="bold")
        for side in ins.spines.values():
            side.set_color(MUTED)
        ax.indicate_inset_zoom(ins, edgecolor=MUTED)

    header(fig, "Круг напряжений: на скорости места для тока не остаётся",
           f"Модель moteus, питание {bus:.2f} В, ток {i:.2f} A, Kv 339, r_max 1.00; "
           f"радиус круга {ve:.2f} В")
    footer(fig, "PNDLM-12 · R, L, p, m — record r_max.inputs · "
                "measurements/2026-11-04-torque-speed.json")
    layout(fig)
    fig.savefig(target, dpi=160)
    print(f"wrote {target}")


def main(source: Path, target_ts: Path, target_circle: Path) -> None:
    record = json.loads(source.read_text())
    inputs = record["r_max"]["inputs"]
    kt = inputs["kt_nm_per_a"]
    new = Model(record["r_max"]["kv_from_measured_kt"]["kv"], 1.0, inputs)
    old = Model(OLD_KV, OLD_R_MAX, inputs)
    torque_speed(record, new, old, kt, target_ts)
    voltage_circle(new, kt, target_circle)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
