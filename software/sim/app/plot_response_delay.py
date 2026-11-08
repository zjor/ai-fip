"""Plot the actuator response-delay figures for the PNDLM-8 article draft.

Usage, from software/sim:

    poetry run python -m app.plot_response_delay pulse \
        ../../docs/hardware/measurements/20261025-response-timing-0020-telemetry.csv \
        ../../docs/hardware/measurements/20261025-response-timing-0020.json \
        0 ../../docs/content/figures/2026-11-08-response-delay-pulse.png

    poetry run python -m app.plot_response_delay scale \
        ../../docs/content/figures/2026-11-08-response-delay-scale.png

``pulse`` (figure A): one torque pulse against time from the host request,
with every host sample, the noise thresholds and the detection times from the
derived analysis JSON. The telemetry CSV is a column extract of the git-ignored
raw ``telemetry.csv`` of run ``20261025T115034Z_response-timing`` (all 330
rows); the JSON records the raw file's SHA-256.

``scale`` (figure B): measured delay bounds next to the requirement, the
simulated LQR limit, textbook limits and the fall time. Values and sources are
listed in ``SCALE_ROWS``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C_POS, C_NEG, BAND = "#2a78d6", "#eb6834", "#cde2fb"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_pulse(telemetry: Path, analysis_json: Path, trial: int, out: str) -> None:
    analysis = json.load(open(analysis_json))
    run_id = analysis["source"]["run_id"]
    result = next(t for t in analysis["trials"] if t["trial_id"] == trial)
    rows = [r for r in csv.DictReader(open(telemetry))
            if int(r["trial_id"]) == trial]
    pulse = next(r for r in rows if r["experiment_phase"] == "pulse")
    t0 = float(pulse["request_time_s"])
    torque_cmd = float(pulse["feedforward_torque_nm"])
    timeout_ms = float(pulse["watchdog_timeout_s"]) * 1000

    def t_ms(r):
        return (float(r["response_time_s"]) - t0) * 1000

    window = [r for r in rows if -20 <= t_ms(r) <= 45]
    sig = result["signals"]
    cur, pos = sig["current_a"], sig["position_rev"]
    first = cur["first_response_ms"]
    confirm = cur["confirmation_response_ms"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, facecolor=SURFACE)
    for ax in axes:
        _style(ax)
        ax.axvspan(0, timeout_ms, color=BAND, alpha=0.45, lw=0)
        for x in (first, confirm):
            ax.axvline(x, color=C_POS, lw=1.2, ls=":")

    # 1. Torque: command vs what the controller reports.
    ax = axes[0]
    ax.step([-20, 0, timeout_ms, 45], [0, torque_cmd * 1000, 0, 0], where="post",
            color=MUTED, lw=2, ls="--", label="команда хоста")
    ax.plot([t_ms(r) for r in window], [float(r["torque_nm"]) * 1000 for r in window],
            color=INK, lw=1.5, marker="o", ms=5, label="момент по данным драйвера")
    ax.set_ylabel("момент, mNm", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    # 2. q-current change from the at-rest median; mode 11 is after the watchdog.
    ax = axes[1]
    active = [r for r in window if r["mode"] != "11"]
    after = [r for r in window if r["mode"] == "11"]
    ax.axhspan(-cur["threshold"], cur["threshold"], color=GRID, lw=0)
    ax.plot([t_ms(r) for r in active],
            [float(r["q_current_a"]) - cur["baseline"] for r in active],
            color=INK, lw=1.5, marker="o", ms=5)
    ax.scatter([t_ms(r) for r in after],
               [float(r["q_current_a"]) - cur["baseline"] for r in after],
               s=26, facecolor=SURFACE, edgecolor=MUTED, lw=1.2)
    ax.set_ylabel("q-ток − покой, A", color=INK)
    ax.text(-19, cur["threshold"], f"шум в покое: порог ±{cur['threshold']:.2f} A",
            va="bottom", fontsize=9, color=MUTED)
    if after:
        ax.text(t_ms(after[0]) + 1.5, cur["threshold"] + 0.05,
                "○ после watchdog: не в анализе", fontsize=9, color=MUTED)

    # 3. Rotor angle change from the at-rest median.
    ax = axes[2]
    thr_deg = pos["threshold"] * 360
    ax.axhspan(-thr_deg, thr_deg, color=GRID, lw=0)
    ax.plot([t_ms(r) for r in window],
            [(float(r["position_rev"]) - pos["baseline"]) * 360 for r in window],
            color=INK, lw=1.5, marker="o", ms=5)
    ax.set_ylabel("угол − покой, °", color=INK)
    ax.text(-19, thr_deg, f"порог ±{thr_deg:.2f}°", va="bottom", fontsize=9, color=MUTED)
    ax.text(timeout_ms / 2, ax.get_ylim()[1] * 0.85,
            f"импульс {timeout_ms:.0f} мс,\nснимает watchdog",
            ha="center", va="top", fontsize=9, color=MUTED)
    ax.set_xlabel("время от команды хоста, мс (точки — замеры хоста, ~5.5 мс)", color=INK)
    ax.set_xlim(-20, 45)

    top = axes[0]
    ymax = top.get_ylim()[1]
    top.set_ylim(top.get_ylim()[0], ymax * 1.25)
    top.text(first - 0.6, ymax * 1.18, f"первый отклик\n{first:.1f} мс",
             ha="right", va="top", fontsize=9, color=INK)
    top.text(confirm + 0.6, ymax * 1.18, f"подтверждение\n{confirm:.1f} мс",
             ha="left", va="top", fontsize=9, color=INK)

    fig.suptitle(f"Импульс {torque_cmd * 1000:+.0f} mNm: ток виден на первом замере после команды",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.text(0.02, 0.005, f"Прогон {run_id}, импульс {trial}. "
             f"Источник: {telemetry.name}, {analysis_json.name}.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out, dpi=160, facecolor=SURFACE)


# (label, start ms, end ms, kind, value label). Sources:
#   PNDLM-9: docs/hardware/specs/motor-torque-fixture.md §9.3
#   PNDLM-8: docs/hardware/desk-stand-torque-pulse-timing.md
#   requirement, LQR sweep, t_c: docs/physics/phase-0-feasibility.md
#     ("Preliminary sensor timing budget", re-run 2026-11-08)
#   theory: λ = 1/139 ms; λ·d < 0.5 (Skogestad) and λ·d < √2 (Stépán, PD)
SCALE_ROWS = [
    ("Измерено, PNDLM-9 (~385 Hz)", 0, 4.4, "measured", "≤ 4.4"),
    ("Измерено, PNDLM-8 (183 Hz)", 0, 13.0, "measured", "≤ 13"),
    ("Допустимая задержка (10 % от t_c)", 0, 13.9, "limit", "13.9"),
    ("Симуляция, текущий LQR: удерживает", 0, 15.0, "limit", "15"),
    ("Симуляция, текущий LQR: не удерживает", 20.0, 220.0, "fail", "от 20"),
    ("Теория: правило робастности, λd < 0.5", 0, 69.5, "limit", "70"),
    ("Теория: без предсказания, λd < √2", 0, 196.6, "limit", "197"),
    ("Время падения t_c", 139.0, 161.0, "fall", "139–161"),
]


def plot_scale(out: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    colors = {"measured": C_POS, "limit": MUTED, "fail": C_NEG, "fall": INK}
    for i, (label, start, end, kind, value) in enumerate(SCALE_ROWS):
        y = len(SCALE_ROWS) - 1 - i
        ax.barh(y, end - start, left=start, height=0.56, color=colors[kind],
                alpha=0.35 if kind == "fail" else 1.0, lw=0)
        if kind == "fail":
            ax.text(start + 2, y, f"{value} мс", va="center", fontsize=9, color=INK)
        else:
            ax.text(end + 2, y, f"{value} мс", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(SCALE_ROWS)))
    ax.set_yticklabels([r[0] for r in reversed(SCALE_ROWS)], color=INK, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 220)
    ax.set_xlabel("мс", color=INK)
    ax.set_title("Задержка привода против времени падения маятника",
                 color=INK, fontsize=12, loc="left")
    fig.text(0.01, 0.01, "Худший вариант конструкции (CAD, t_c = 139 мс). "
             "Измерено на ноутбуке + fdcanusb; теория — для линейной модели.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=160, facecolor=SURFACE)


def main() -> None:
    if len(sys.argv) == 6 and sys.argv[1] == "pulse":
        plot_pulse(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    elif len(sys.argv) == 3 and sys.argv[1] == "scale":
        plot_scale(sys.argv[2])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
