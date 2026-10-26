"""Plot the static torque-constant result for an article or video.

Usage, from software/sim:

    poetry run python -m app.plot_torque_constant \
        ../../docs/hardware/measurements/2026-10-26-torque-constant.json \
        ../../docs/content/figures/2026-10-26-torque-constant.png

Left: physical torque on the scale against commanded q-current, with the
firmware prediction (8.27/Kv) and the measured linear-region line. Right: the
residual after subtracting the measured line, which shows the seating zone
below 2 A and the constant offset between the two sides.
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C_POS, C_NEG, BAND = "#2a78d6", "#eb6834", "#cde2fb"


def main() -> None:
    d = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    kt_fw = d["simulator"]["kt_firmware_nm_per_a"]
    kt = d["simulator"]["kt_linear_region_per_commanded_a_nm_per_a"]
    plateaus = [p for p in d["plateaus"] if p["physical_torque_nm"] is not None]
    sides = (("positive", C_POS, "o", "+ сторона"), ("negative", C_NEG, "s", "− сторона"))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=SURFACE)
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)
        a.grid(color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(MUTED)
        a.tick_params(colors=MUTED, labelsize=9)
        a.set_xlabel("командный q-ток, A", color=INK)

    xs = [0, 6.3]
    ax.plot(xs, [kt_fw * x * 1000 for x in xs], color=MUTED, lw=2, ls="--",
            label=f"прогноз прошивки: 8.27/Kv = {kt_fw * 1000:.1f} mNm/A")
    ax.plot(xs, [kt * x * 1000 for x in xs], color=INK, lw=2,
            label=f"измерено (2–6 A, обе стороны): {kt * 1000:.1f} mNm/A")
    for direction, color, marker, label in sides:
        mine = [p for p in plateaus if p["direction"] == direction]
        ax.scatter([abs(p["command_a"]) for p in mine],
                   [abs(p["physical_torque_nm"]) * 1000 for p in mine],
                   s=46, color=color, edgecolor=SURFACE, lw=1.2, marker=marker,
                   label=f"{label}, весы", zorder=3)
        ax2.scatter([abs(p["command_a"]) for p in mine],
                    [(abs(p["physical_torque_nm"]) - kt * abs(p["command_a"])) * 1000 for p in mine],
                    s=46, color=color, edgecolor=SURFACE, lw=1.2, marker=marker, label=label, zorder=3)
    ax.set_ylabel("физический момент, mNm", color=INK)
    ax.set_title("Момент на весах vs команда", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.text(6.25, kt_fw * 6.25 * 1000 - 2, f"{100 * (kt / kt_fw - 1):+.1f} %", color=MUTED,
            fontsize=9, ha="right", va="top")

    ax2.axhline(0, color=INK, lw=1.5)
    ax2.axhspan(-5, 5, color=BAND, alpha=0.5, lw=0, label="принятое возмущение ±5 mNm")
    ax2.axvspan(0, 2, color=GRID, alpha=0.6, lw=0, label="зона посадки рычага (< 2 A)")
    ax2.set_ylabel(f"отклонение от прямой {kt * 1000:.1f} mNm/A, mNm", color=INK)
    ax2.set_title("Остаток: что не описывает K_t·I", color=INK, fontsize=12, loc="left")
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.set_xlim(0, 6.3)
    ax2.set_ylim(-16, 8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
