"""Plot the PNDLM-11 rotating-loss figure for the article draft.

Usage, from software/sim:

    poetry run python -m app.plot_rotating_losses \
        ../../docs/hardware/measurements/2026-11-01-rotating-losses.json \
        ../../docs/articles/figures/2026-11-01-rotating-losses.png

Top panel: coast-down loss torque against speed (1 rev/s bins, mean ± sd, both
directions pooled), the pooled fit a + b·ω + c·ω², its windage part c·ω², and
the plateau cross-check (|mean i_q| × K_t) at each ladder step.
Bottom panel: plateau minus coast loss at each step, first analysis
(mean |i_q|, superseded) against the re-analysis (|mean i_q|), with the
coast-fit rmse as a band.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from app.plot_style import (
    MEASURED,
    MEASURED_2,
    MEASURED_BAND,
    MUTED,
    REFERENCE,
    footer,
    header,
    layout,
    plt,
    style,
)

RPM_PER_REV_S = 60.0
RAD_S_PER_RPM = 2.0 * math.pi / 60.0


def main(source: Path, target: Path) -> None:
    record = json.loads(source.read_text())
    model = record["loss_model"]
    fit = model["pooled"]
    bins = model["loss_map_1rev_s_bins"]
    rows = record["plateau_cross_check"]["rows"]

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.0, 8.6), height_ratios=(2.2, 1.0), sharex=True
    )

    rpm = [b["speed_rev_s"] * RPM_PER_REV_S for b in bins]
    tau = [b["tau_mean_nm"] * 1e3 for b in bins]
    sd = [b["tau_sd_nm"] * 1e3 for b in bins]
    top.errorbar(rpm, tau, yerr=sd, fmt="o", ms=4, color=MEASURED, ecolor=MEASURED_BAND,
                 elinewidth=2, label="выбег, бины 1 rev/s (среднее ± sd)")

    grid = [10.0 * k for k in range(0, 301)]
    w = [r * RAD_S_PER_RPM for r in grid]
    top.plot(grid, [(fit["a_nm"] + fit["b_nm_s"] * x + fit["c_nm_s2"] * x * x) * 1e3 for x in w],
             color=MEASURED, lw=2.2,
             label=f"фит a + bω + cω²: a = {fit['a_nm'] * 1e3:.2f} mNm, "
                   f"b = {fit['b_nm_s'] * 1e6:.1f} µNm·s")
    top.plot(grid, [fit["c_nm_s2"] * x * x * 1e3 for x in w], color=REFERENCE, lw=1.2, ls="--",
             label="из них сопротивление воздуха cω²")
    top.plot([r["speed_rpm"] for r in rows], [r["plateau_current_x_kt_nm"] * 1e3 for r in rows],
             "D", ms=6, mfc="none", mec=MEASURED_2, mew=1.6,
             label="плато: |среднее i_q| × K_t")
    top.set_ylabel("момент потерь, mNm")
    top.set_ylim(0, 20)
    top.legend(loc="upper left", frameon=False)
    style(top)

    old = [(r["plateau_mean_abs_x_kt_nm_superseded"] - r["coast_loss_effective_basis_nm"]) * 1e3
           for r in rows]
    new = [r["gap_nm"] * 1e3 for r in rows]
    speeds = [r["speed_rpm"] for r in rows]
    rmse = fit["rmse_nm"] * 1e3
    bottom.axhspan(-rmse, rmse, color=MEASURED_BAND, alpha=0.6, lw=0,
                   label=f"± rmse фита выбега ({rmse:.2f} mNm)")
    bottom.axhline(0, color=MUTED, lw=0.8)
    bottom.plot(speeds, old, "o", ms=5, mfc="none", mec=REFERENCE, mew=1.2,
                label="первый анализ: среднее |i_q|")
    bottom.plot(speeds, new, "D", ms=6, color=MEASURED_2, label="пересчёт: |среднее i_q|")
    bottom.set_ylabel("плато − выбег, mNm")
    bottom.set_xlabel("скорость, rpm")
    bottom.set_xlim(0, 3100)
    bottom.set_ylim(-1.0, 2.2)
    bottom.legend(loc="upper left", frameon=False, ncol=1, fontsize=10)
    style(bottom)

    header(fig, "Потери мотора: 6 mNm на 500 rpm, 17 mNm на 3000 rpm",
           "Свободный выбег тестового диска, обе стороны вместе; внизу — сверка с током плато")
    footer(fig, "PNDLM-11 · 4 прогона, 28.10 и 01.11.2026, 505 окон · "
                "measurements/2026-11-01-rotating-losses.json")
    layout(fig)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    print(f"wrote {target}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
